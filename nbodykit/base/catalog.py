from six import add_metaclass, string_types
from nbodykit.transform import ConstantArray
from nbodykit import _globals

import abc
import numpy
import logging
import warnings
from collections import defaultdict
import dask.array as da

def get_slice_size(start, stop, step):
    """
    Utility function to return the size of an array slice
    """
    if step is None: step = 1
    if start is None: start = 0
    N, remainder = divmod(stop-start, step)
    if remainder: N += 1
    return N

def isgetitem(task):
    """
    Return True if the task is a getitem slice
    """
    from dask.array.optimization import GETTERS

    if isinstance(task, tuple) and task[0] in GETTERS:
        sl = task[2][0] # first dimension slice
        if sl is not None:
            return True
    return False

def find_consecutive_gets(t, out, dsk, dependents):
    """
    Fill the defaultdict ``out`` with consecutive getitem tasks that have
    only (at most) a single dependent (no split chains).

    Keys are the block number along the first index for each task.
    """
    # check if task is getitem with only 1 dependent
    if isgetitem(dsk[t]) and len(dependents[t]) <= 1:
        out[t[1]].append(t)
        for xx in dependents[t]:
            find_consecutive_gets(xx, out, dsk, dependents)


def expanding_apply(iterable, func):
    """
    Perform a running application of ``func`` to ``iterable``
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    for element in it:
        total = func(total, element)
    return total

def optimized_selection(arr, index):
    """
    Perform an optimized selection operation on the input dask array ``arr``,
    as specified by the boolean index ``index``.

    The operation is optimized by inserting slicing tasks directly after
    the task that last alters the length of the array (along axis=0).

    The procedure used is as follows:

    - Identify the source nodes in the task graph (with no dependencies)
    - Following each source node, identify the number of consecutive getitem
      tasks
    - Compute the size of the fused getitem slices -- if the size is equal to
      the size of the catalog on the rank, insert the necessary slicing
      tasks following the last getitem

    Parameters
    ----------
    arr : dask.array.Array
        the dask array that we wish to slice
    index : array_like
        a boolean array representing the selection indices; should be
        the same length as ``arr``
    """
    from dask.array.optimization import fuse_slice, cull, reverse_dict
    from dask.array.slicing import slice_array, tokenize
    from dask.core import subs

    # need the "catalog" attribute
    assert isinstance(arr, ColumnAccessor)

    # cull unused tasks first
    dsk, dependencies = cull(arr.dask, arr._keys())

    # the dependents
    dependents = reverse_dict(dependencies)

    # this store the new slicing tasks
    dsk2 = dict()

    # source nodes in graph have no dependencies
    sources = [k for k in dependencies if not len(dependencies[k])]

    name = arr.name

    # loop over all of the source tasks
    for source in sources:

        # find consecutive getitems from this source
        gettasks = defaultdict(list)
        for dep in dependents[source]:
            find_consecutive_gets(dep, gettasks, dsk, dependents)

        # compute total size of getitem slices
        size = 0
        for chunknum in gettasks:
            keys = iter(gettasks[chunknum])
            slices = []
            for kk in keys:
                v = dsk[kk][2][0] # the slice along the 1st axis

                # just save slices
                if isinstance(v, slice):
                    slices.append(v)
                # if array_like, the array gives indices of valid elements
                elif isinstance(v, (numpy.ndarray,list)):
                    dummy_slice = slice(0,len(v),None) # dummy slice of the right final size
                    slices.append(dummy_slice)
                else:
                    raise ValueError("cannot perform optimized selection")

            # fuse all of the slices together and determined size of fused slice
            total_slice = expanding_apply(slices, fuse_slice)

            # try to identify the stop from previous data
            if total_slice.stop is None:
                N = len(arr) # if all gets end in None, use length of array
                total_slice = slice(total_slice.start, N, total_slice.step)

            # get the size
            size += get_slice_size(total_slice.start, total_slice.stop, total_slice.step)

        # if no getter tasks, size must be length of array
        if not len(gettasks):
            size = len(arr)
            input_task = source
        else:
            input_task = gettasks[0][-1] # last getitem task key along block #0

        # "input_task" is the task input into the slicing graph, key must be a tuple
        if not isinstance(input_task, tuple):
            raise ValueError("optimized selection failure: input task key is not a tuple %s" %str(input_task))

        # total slice size must be equal to catalog size to work
        if arr.catalog.size == size:

            # need all blocks along 1st axis to be represented
            if len(gettasks) and not all(block in gettasks for block in range(arr.numblocks[0])):
                raise ValueError("optimized selection failure: missing blocks")

            # determine the slice tasks
            # output task name of slice graph is "selection-*"
            # input task name of slice graph is last consecutive getitem task
            ndim = len(input_task)-1 # dimensions of the chunks
            inname = input_task[0] # input task name
            outname = 'selection-'+tokenize(arr,index,source)
            slice_dsk, blockdims = slice_array(outname, inname, arr.chunks[:ndim], (index,))

            # if last getitem task is array name, we need to rename array
            if inname == arr.name:
                name = outname

            # add the slice tasks to the new graph
            dsk2.update(slice_dsk)

            # update dependents of last consecutive getitem task
            # to point to "selection-*" tasks
            for k,v in slice_dsk.items():
                old_task_key = v[1]
                for dep in dependents[old_task_key]:
                    dsk2[dep] = subs(dsk[dep], old_task_key, k)

    # if no new tasks, then we failed to verify size or find sources
    if not len(dsk2):
        raise ValueError("cannot perform optimized selection")

    # update the original graph and make new Array
    dsk.update(dsk2)
    chunks = tuple(blockdims) + arr.chunks[ndim:]
    return da.Array(dsk, name, chunks, dtype=arr.dtype)


class ColumnAccessor(da.Array):
    """
    Provides access to a Column from a Catalog

    This is a thin subclass of :class:`dask.array.Array` to
    provide a reference to the catalog object,
    an additional ``attrs`` attribute (for recording the
    reproducible meta-data), and some pretty print support.

    Due to particularity of :mod:`dask`, any transformation
    that is not explicitly in-place will return
    a :class:`dask.array.Array`, and losing the pointer to
    the original catalog and the meta data attrs.
    """
    def __new__(cls, catalog, daskarray):
        self = da.Array.__new__(ColumnAccessor,
                daskarray.dask,
                daskarray.name,
                daskarray.chunks,
                daskarray.dtype,
                daskarray.shape)
        self.catalog = catalog
        self.attrs = {}
        return self

    def __getitem__(self, key):
        """
        If ``key`` is an array-like index with the same length as the
        underlying catalog, the slice will be performed with the
        optimization provided by :func:`optimized_selection`.

        Otherwise, the default behavior is returned.
        """
        d = None

        # try to optimize the selection
        sel = key[0] if isinstance(key, tuple) else key
        if isinstance(sel, (list, numpy.ndarray, da.Array)):
            if len(sel) == self.catalog.size:
                if isinstance(sel, da.Array):
                    sel = self.catalog.compute(sel)
                try:
                    d = optimized_selection(self, sel)
                except:
                    pass

        # the fallback is default behavior
        if d is None:
            d = da.Array.__getitem__(self, key)

        # return a ColumnAccessor (okay b/c __setitem__ checks for circular references)
        toret = ColumnAccessor(self.catalog, d)
        toret.attrs.update(self.attrs)
        return toret

    def as_daskarray(self):
        return da.Array(
                self.dask,
                self.name,
                self.chunks,
                self.dtype,
                self.shape)

    def compute(self):
        return self.catalog.compute(self)

    def __str__(self):
        r = da.Array.__str__(self)
        if len(self) > 0:
            r = r + " first : %s" % str(self[0].compute())
        if len(self) > 1:
            r = r + " last: %s" % str(self[-1].compute())
        return r

def column(name=None):
    """
    Decorator that defines a function as a column in a CatalogSource
    """
    def decorator(getter):
        getter.column_name = name
        return getter

    if hasattr(name, '__call__'):
        # a single callable is provided
        getter = name
        name = getter.__name__
        return decorator(getter)
    else:
        return decorator


def find_columns(cls):
    """
    Find all hard-coded column names associated with the input class

    Returns
    -------
    hardcolumns : set
        a set of the names of all hard-coded columns for the
        input class ``cls``
    """
    hardcolumns = []

    # search through the class dict for columns
    for key, value in cls.__dict__.items():
         if hasattr(value, 'column_name'):
            hardcolumns.append(value.column_name)

    # recursively search the base classes, too
    for base in cls.__bases__:
        hardcolumns += find_columns(base)

    return list(sorted(set(hardcolumns)))


def find_column(cls, name):
    """
    Find a specific column ``name`` of an input class, or raise
    an exception if it does not exist

    Returns
    -------
    column : callable
        the callable that returns the column data
    """
    # first search through class
    for key, value in cls.__dict__.items():
        if not hasattr(value, 'column_name'): continue
        if value.column_name == name: return value

    for base in cls.__bases__:
        try: return find_column(base, name)
        except: pass

    args = (name, str(cls))
    raise AttributeError("unable to find column '%s' for '%s'" % args)


class CatalogSourceBase(object):
    """
    An abstract base class representing a catalog of discrete particles.

    This objects behaves like a structured numpy array -- it must have a
    well-defined size when initialized. The ``size`` here represents the
    number of particles in the source on the local rank.

    Subclasses of this class must define a ``size`` attribute.

    The information about each particle is stored as a series of
    columns in the format of dask arrays. These columns can be accessed
    in a dict-like fashion.
    """
    logger = logging.getLogger('CatalogSourceBase')

    @staticmethod
    def make_column(array):
        """
        Utility function to convert a numpy array to a :class:`dask.array.Array`.
        """
        if isinstance(array, da.Array):
            return array
        elif isinstance(array, ColumnAccessor):
            # important to get the accessor as a dask array to avoid circular
            # references
            return array.as_daskarray()
        else:
            return da.from_array(array, chunks=_globals['dask_chunk_size'])

    def __init__(self, comm, use_cache=False):

        # ensure self.comm is set, though usually already set by the child.
        self.comm = comm

        # initialize a cache
        self.use_cache = use_cache

        # user-provided overrides for columns
        self._overrides = {}

    def __iter__(self):
        return iter(self.columns)

    def __contains__(self, col):
        return col in self.columns

    def __getitem__(self, sel):
        """
        The following types of indexing are supported:

        #.  strings specifying a column in the CatalogSource; returns
            a dask array holding the column data
        #.  boolean arrays specifying a slice of the CatalogSource;
            returns a CatalogSource holding only the revelant slice
        #.  slice object specifying which particles to select
        #.  list of strings specifying column names; returns a CatalogSource
            holding only the selected columnss
        """
        # handle boolean array slices
        if not isinstance(sel, string_types):

            # convert slices to boolean arrays
            if isinstance(sel, (slice, list)):

                # select a subset of list of string column names
                if isinstance(sel, list) and all(isinstance(ss, string_types) for ss in sel):
                    invalid = set(sel) - set(self.columns)
                    if len(invalid):
                        msg = "cannot select subset of columns from "
                        msg += "CatalogSource due to invalid columns: %s" % str(invalid)
                        raise KeyError(msg)

                    # return a CatalogSource only holding the selected columns
                    subset_data = {col:self[col] for col in sel}
                    toret = CatalogSource._from_columns(self.size, self.comm, use_cache=self.use_cache, **subset_data)
                    toret.attrs.update(self.attrs)
                    return toret

                # list must be all integers
                if isinstance(sel, list) and not numpy.array(sel).dtype == numpy.integer:
                    raise KeyError("array like indexing via a list should be a list of integers")

                # size must be implemented
                if self.size is NotImplemented:
                    raise ValueError("cannot perform selection due to NotImplemented size")

                # convert into slice into boolean array
                index = numpy.zeros(self.size, dtype='?')
                index[sel] = True; sel = index

            # do the slicing
            if not numpy.isscalar(sel):
                return get_catalog_subset(self, sel)
            else:
                raise KeyError("strings and boolean arrays are the only supported indexing methods")

        # get the right column
        if sel in self._overrides:
            r = self._overrides[sel]
        elif sel in self.hardcolumns:
            r = self.get_hardcolumn(sel)
        else:
            raise KeyError("column `%s` is not defined in this source; " %sel + \
                            "try adding column via `source[column] = data`")

        # evaluate callables if we need to
        if callable(r): r = r()

        r = ColumnAccessor(self, r)

        return r

    def __setitem__(self, col, value):
        """
        Add new columns to the CatalogSource, overriding any existing columns
        with the name ``col``.
        """
        self._overrides[col] = self.make_column(value)

    @property
    def use_cache(self):
        """
        If set to ``True``, use the built-in caching features of ``dask``
        to cache data in memory.
        """
        return self._use_cache

    @use_cache.setter
    def use_cache(self, val):
        """
        Initialize a Cache object of size set by the ``dask_cache_size``
        global configuration option, which is 1 GB by default.
        """
        if val:
            try:
                from dask.cache import Cache
                if not hasattr(self, '_cache'):
                    self._cache = Cache(_globals['dask_cache_size'])
            except ImportError:
                warnings.warn("caching of CatalogSource requires ``cachey`` module; turning cache off")
        else:
            if hasattr(self, '_cache'): delattr(self, '_cache')
        self._use_cache = val

    @property
    def hardcolumns(self):
        """
        A list of the hard-coded columns in the CatalogSource.

        These columns are usually member functions marked by @column decorator.
        Subclasses may override this method and use :func:`get_hardcolumn` to
        bypass the decorator logic.
        """
        try:
            self._hardcolumns
        except AttributeError:
            self._hardcolumns = find_columns(self.__class__)
        return self._hardcolumns

    @property
    def columns(self):
        """
        All columns in the CatalogSource, including those hard-coded into
        the class's defintion and override columns provided by the user.
        """
        hcolumns  = list(self.hardcolumns)
        overrides = list(self._overrides)
        return sorted(set(hcolumns + overrides))

    @property
    def attrs(self):
        """
        A dictionary storing relevant meta-data about the CatalogSource.
        """
        try:
            return self._attrs
        except AttributeError:
            self._attrs = {}
            return self._attrs

    def get_hardcolumn(self, col):
        """
        Construct and return a hard-coded column.

        These are usually produced by calling member functions marked by the
        @column decorator.

        Subclasses may override this method and the hardcolumns attribute to
        bypass the decorator logic.
        """
        return find_column(self.__class__, col)(self)

    def compute(self, *args, **kwargs):
        """
        Our version of :func:`dask.compute` that computes
        multiple delayed dask collections at once.

        This should be called on the return value of :func:`read`
        to converts any dask arrays to numpy arrays.

        If :attr:`use_cache` is ``True``, this internally caches data, using
        dask's built-in cache features.

        Parameters
        -----------
        args : object
            Any number of objects. If the object is a dask
            collection, it's computed and the result is returned.
            Otherwise it's passed through unchanged.

        Notes
        -----
        The dask default optimizer induces too many (unnecesarry)
        IO calls -- we turn this off feature off by default. Eventually we
        want our own optimizer probably.
        """
        import dask

        # do not optimize graph (can lead to slower optimizations)
        kwargs.setdefault('optimize_graph', False)

        # use a cache?
        if self.use_cache and hasattr(self, '_cache'):
            with self._cache:
                toret = dask.compute(*args, **kwargs)
        else:
            toret = dask.compute(*args, **kwargs)

        # do not return tuples of length one
        if len(toret) == 1: toret = toret[0]
        return toret

    def save(self, output, columns, datasets=None, header='Header'):
        """
        Save the CatalogSource to a :class:`bigfile.BigFile`.

        Only the selected columns are saved and :attr:`attrs` are saved in
        ``header``. The attrs of columns are stored in the datasets.

        Parameters
        ----------
        output : str
            the name of the file to write to
        columns : list of str
            the names of the columns to save in the file
        datasets : list of str, optional
            names for the data set where each column is stored; defaults to
            the name of the column
        header : str, optional
            the name of the data set holding the header information, where
            :attr:`attrs` is stored
        """
        import bigfile
        import json
        from nbodykit.utils import JSONEncoder

        if datasets is None: datasets = columns
        if len(datasets) != len(columns):
            raise ValueError("`datasets` must have the same length as `columns`")

        with bigfile.BigFileMPI(comm=self.comm, filename=output, create=True) as ff:
            try:
                bb = ff.open(header)
            except:
                bb = ff.create(header)
            with bb :
                for key in self.attrs:
                    try:
                        bb.attrs[key] = self.attrs[key]
                    except ValueError:
                        try:
                            json_str = 'json://'+json.dumps(self.attrs[key], cls=JSONEncoder)
                            bb.attrs[key] = json_str
                        except:
                            raise ValueError("cannot save '%s' key in attrs dictionary" % key)

            for column, dataset in zip(columns, datasets):
                c = self[column]

                # save column attrs too
                with ff.create_from_array(dataset, c) as bb:
                    if hasattr(c, 'attrs'):
                        for key in c.attrs:
                            bb.attrs[key] = c.attrs[key]

    def read(self, columns):
        """
        Return the requested columns as dask arrays.

        Parameters
        ----------
        columns : list of str
            the names of the requested columns

        Returns
        -------
        list of :class:`dask.array.Array` :
            the list of column data, in the form of dask arrays
        """
        missing = set(columns) - set(self.columns)
        if len(missing) > 0:
            msg = "source does not contain columns: %s; " %str(missing)
            msg += "try adding columns via `source[column] = data`"
            raise ValueError(msg)

        return [self[col] for col in columns]

    def to_mesh(self, Nmesh=None, BoxSize=None, dtype='f4', interlaced=False,
                compensated=False, window='cic', weight='Weight',
                value='Value', selection='Selection', position='Position'):
        """
        Convert the CatalogSource to a MeshSource, using the specified
        parameters.

        Parameters
        ----------
        Nmesh : int, optional
            the number of cells per side on the mesh; must be provided if
            not stored in :attr:`attrs`
        BoxSize : scalar, 3-vector, optional
            the size of the box; must be provided if
            not stored in :attr:`attrs`
        dtype : string, optional
            the data type of the mesh array
        interlaced : bool, optional
            use the interlacing technique of Sefusatti et al. 2015 to reduce
            the effects of aliasing on Fourier space quantities computed
            from the mesh
        compensated : bool, optional
            whether to correct for the window introduced by the grid
            interpolation scheme
        window : str, optional
            the string specifying which window interpolation scheme to use;
            see `pmesh.window.methods`
        weight : str, optional
            the name of the column specifying the weight for each particle
        value: str, optional
            the name of the column specifying the field value for each particle
        selection : str, optional
            the name of the column that specifies which (if any) slice
            of the CatalogSource to take
        position : str, optional
            the name of the column that specifies the position data of the
            objects in the catalog

        Returns
        -------
        mesh : CatalogMesh
            a mesh object that provides an interface for gridding particle
            data onto a specified mesh
        """
        from nbodykit.base.catalogmesh import CatalogMesh
        from pmesh.window import methods

        # make sure all of the columns exist
        for col in [weight, selection]:
            if col not in self:
                raise ValueError("column '%s' missing; cannot create mesh" %col)

        if window not in methods:
            raise ValueError("valid window methods: %s" %str(methods))

        if BoxSize is None:
            try:
                BoxSize = self.attrs['BoxSize']
            except KeyError:
                raise ValueError(("cannot convert particle source to a mesh; "
                                  "'BoxSize' keyword is not supplied and the CatalogSource "
                                  "does not define one in 'attrs'."))
        if Nmesh is None:
            try:
                Nmesh = self.attrs['Nmesh']
            except KeyError:
                raise ValueError(("cannot convert particle source to a mesh; "
                                  "'Nmesh' keyword is not supplied and the CatalogSource "
                                  "does not define one in 'attrs'."))

        r = CatalogMesh(self, Nmesh=Nmesh, BoxSize=BoxSize, dtype=dtype,
                                weight=weight, selection=selection,
                                value=value, position=position)
        r.interlaced = interlaced
        r.compensated = compensated
        r.window = window
        return r


@add_metaclass(abc.ABCMeta)
class CatalogSource(CatalogSourceBase):
    """
    An abstract base class representing a catalog of discrete particles.

    This objects behaves like a structured numpy array -- it must have a
    well-defined size when initialized. The ``size`` here represents the
    number of particles in the source on the local rank.

    Subclasses of this class must define a ``size`` attribute.

    The information about each particle is stored as a series of
    columns in the format of dask arrays. These columns can be accessed
    in a dict-like fashion.

    All subclasses of this class contain the following default columns:

    #. ``Weight``
    #. ``Value``
    #. ``Selection``

    For a full description of these default columns, see
    :ref:`the documentation <catalog-source-default-columns>`.
    """
    logger = logging.getLogger('CatalogSource')

    @classmethod
    def _from_columns(kls, size, comm, use_cache=False, **columns):
        """ Create a Catalog from a set of columns.

            This method is used internally by nbodykit to create
            views of catalogs based on existing catalogs.

            The attrs attribute of the returned catalog is empty.

            Use :class:`~nbodykit.source.catalog.array.ArrayCatalog`
            To adapt a structured array or dictionary of array.

        """
        self = object.__new__(CatalogSource)

        self._size = size
        CatalogSource.__init__(self, comm=comm, use_cache=use_cache)

        # store the column arrays
        for name in columns:
            self[name] = columns[name]

        return self

    def __init__(self, comm, use_cache=False):

        # init the base class
        CatalogSourceBase.__init__(self, comm, use_cache=use_cache)

        # if size is already computed update csize
        # otherwise the subclass shall call update_csize explicitly.
        if self.size is not NotImplemented:
            self.update_csize()

    def __repr__(self):
        size = "%d" %self.size if self.size is not NotImplemented else "NotImplemented"
        return "%s(size=%s)" %(self.__class__.__name__, size)

    def __len__(self):
        """
        The local size of the CatalogSource on a given rank.
        """
        return self.size

    def __setitem__(self, col, value):
        """
        Add columns to the CatalogSource, overriding any existing columns
        with the name ``col``.
        """
        # handle scalar values
        if numpy.isscalar(value):
            assert self.size is not NotImplemented, "size is not implemented! cannot set scalar array"
            value = ConstantArray(value, self.size, chunks=_globals['dask_chunk_size'])

        # check the correct size, if we know the size
        if self.size is not NotImplemented:
            assert len(value) == self.size, "error setting column, data must be array of size %d" %self.size

        # call the base __setitem__
        CatalogSourceBase.__setitem__(self, col, value)

    def copy(self):
        """
        Return a `shallow` copy of the CatalogSource object, where each column is a reference
        of the corresponding column of the ``self``. No copies of data is made.

        Returns
        -------
        CatalogSource:
            the new CatalogSource object holding all of the the data columns of ``self``
        """
        if self.size is NotImplemented:
            return ValueError("cannot copy a CatalogSource that does not have `size` implemented")

        data = {col:self[col] for col in self.columns}
        toret = CatalogSource._from_columns(self.size, comm=self.comm, use_cache=self.use_cache, **data)
        toret.attrs.update(self.attrs)
        return toret

    @property
    def size(self):
        """
        The number of particles in the CatalogSource on the local rank.

        This property must be defined for all subclasses.
        """
        if not hasattr(self, '_size'):
            return NotImplemented
        return self._size

    @size.setter
    def size(self, value):
        raise RuntimeError("Property size is read-only. Internally, _size can be set during Catalog initialization.")

    @property
    def csize(self):
        """
        The total, collective size of the CatalogSource, i.e., summed across all
        ranks.

        It is the sum of :attr:`size` across all available ranks.
        """
        return self._csize

    @column
    def Selection(self):
        """
        A boolean column that selects a subset slice of the CatalogSource.

        By default, this column is set to ``True`` for all particles.
        """
        return ConstantArray(True, self.size, chunks=_globals['dask_chunk_size'])

    @column
    def Weight(self):
        """
        The column giving the weight to use for each particle on the mesh.

        The mesh field is a weighted average of ``Value``, with the weights
        given by ``Weight``.

        By default, this array is set to unity for all particles.
        """
        return ConstantArray(1.0, self.size, chunks=_globals['dask_chunk_size'])

    @column
    def Value(self):
        """
        When interpolating a CatalogSource on to a mesh, the value of this
        array is used as the Value that each particle contributes to a given
        mesh cell.

        The mesh field is a weighted average of ``Value``, with the weights
        given by ``Weight``.

        By default, this array is set to unity for all particles.
        """
        return ConstantArray(1.0, self.size, chunks=_globals['dask_chunk_size'])

    def update_csize(self):
        """
        Set the collective size, :attr:`csize`.

        This function should be called in :func:`__init__` of a subclass,
        after :attr:`size` has been set to a valid value (not ``NotImplemented``)
        """
        if self.size is NotImplemented:
            raise ValueError(("``size`` cannot be NotImplemented when trying "
                              "to compute collective size `csize`"))

        # sum size across all ranks
        self._csize = self.comm.allreduce(self.size)

        # log some info
        if self.comm.rank == 0:
            self.logger.debug("rank 0, local number of particles = %d" %self.size)
            self.logger.info("total number of particles in %s = %d" %(str(self), self.csize))


def get_catalog_subset(parent, index):
    """
    Select a subset of a :class:`CatalogSource` according to a boolean
    index array.

    Returns a :class:`CatalogSource` holding only the data that satisfies
    the slice criterion.

    Parameters
    ----------
    parent : :class:`CatalogSource`
        the parent source that will be sliced
    index : array_like
        either a dask or numpy boolean array; this determines which
        rows are included in the returned object

    Returns
    -------
    subset : :class:`CatalogSource`
        the particle source with the same meta-data as `parent`, and
        with the sliced data arrays
    """
    # compute the index slice if needed and get the size
    if isinstance(index, da.Array):
        index = parent.compute(index)
    elif isinstance(index, list):
        index = numpy.array(index)

    # verify the index is a boolean array
    if len(index) != len(parent):
        raise ValueError("slice index has length %d; should be %d" %(len(index), len(parent)))
    if getattr(index, 'dtype', None) != '?':
        raise ValueError("index used to slice CatalogSource must be boolean and array-like")

    # new size is just number of True entries
    size = index.sum()

    # initialize subset Source of right size
    subset_data = {col:parent[col][index] for col in parent}
    toret = CatalogSource._from_columns(size, parent.comm, use_cache=parent.use_cache, **subset_data)

    # and the meta-data
    toret.attrs.update(parent.attrs)

    return toret
