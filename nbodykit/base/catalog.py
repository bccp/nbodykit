from nbodykit.transform import ConstantArray
from nbodykit import _global_options, CurrentMPIComm

from six import string_types
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

def find_chains(chain, dependents, dsk):
    """
    Walk recursively through a dask graph (a directed graph)
    and find consecutive chains of ``getitem`` tasks.

    This function yields all consecutive chains.
    """
    # chain of keys needs to be a list
    if not isinstance(chain, list):
        chain = [chain]

    # dependents of last key in chain
    deps = dependents[chain[-1]]

    # to be a valid chain, need more than element
    valid = len(chain) > 1

    # each element in chain must be a getter task
    # ignore first element in chain (the source)
    valid &= isgetitem(dsk[chain[-1]])

    # check for any valid dependents
    no_valid_deps = not len(deps) or not any(isgetitem(dsk[dep]) for dep in deps)

    # if chain is valid with no valid deps, it is finished so yield
    if valid and no_valid_deps:
        yield chain

    # check dependents if we are on first path, or if current element is valid
    if len(chain) == 1 or valid:
        for dep in deps:
            new_chain = chain + [dep]
            for chain2 in find_chains(new_chain, dependents, dsk):
                yield chain2

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

class DaskGraphOptimizationFailure(Exception):
    pass

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

    Returns
    -------
    :class:`dask.array.Array` :
        a new array holding the same data with the selection index applied

    Raises
    ------
    DaskGraphOptimizationFailure  :
        if the array size cannot be determined from the task graph and
        the selection index applied properly
    """
    from operator import getitem, itemgetter
    from itertools import groupby
    # catch API changes to DASK
    try:
        from dask.array.optimization import fuse_slice, cull, reverse_dict
        from dask.array.slicing import tokenize
        from dask.core import subs
    except ImportError as e:
        raise DaskGraphOptimizationFailure(e)

    # need the "catalog" attribute
    if not isinstance(arr, ColumnAccessor):
        raise DaskGraphOptimizationFailure("input array must be of type 'ColumnAccessor'")

    # cull unused tasks first
    dsk, dependencies = cull(arr.dask, arr._keys())

    # the dependents
    dependents = reverse_dict(dependencies)

    # this stores the new slicing tasks
    dsk2 = dict()

    # source nodes in graph have no dependencies
    sources = [k for k in dependencies if not len(dependencies[k])]

    # chunk sizes along the first dimension
    chunk_slices = numpy.append([0],numpy.cumsum(arr.chunks[0]))
    chunks = {}

    # sum up boolean array to get new total size
    new_size = int(numpy.sum(index))

    # loop over all of the source tasks
    name = arr.name
    for source in sources:

        # initialize the size of blocks to zero
        blocksizes = {k:0 for k in range(arr.numblocks[0])}

        # find consecutive getitems from this source
        size = 0
        chains = []
        for chain in find_chains(source, dependents, dsk):
            chains.append(chain)

            # extract the slices from the chain
            slices = []
            for kk in chain[1:]: # first object in chain is "source"
                v = dsk[kk][2][0] # the slice along the 1st axis

                # just save slices
                if isinstance(v, slice):
                    slices.append(v)
                # if array_like, the array gives indices of valid elements
                elif isinstance(v, (numpy.ndarray,list)):
                    dummy_slice = slice(0,len(v),None) # dummy slice of the right final size
                    slices.append(dummy_slice)
                else:
                    raise DaskGraphOptimizationFailure("unknown fancy indexing found in graph")

            # fuse all of the slices together and determined size of fused slice
            total_slice = expanding_apply(slices, fuse_slice)

            # try to identify the stop from previous data
            if total_slice.stop is None:
                N = len(arr) # if all gets end in None, use length of array
                total_slice = slice(total_slice.start, N, total_slice.step)

            # get the size
            size += get_slice_size(total_slice.start, total_slice.stop, total_slice.step)

        # if no getter tasks, skip this source
        if not len(chains):
            continue

        # total slice size must be equal to catalog size to work
        if arr.catalog.size == size:

            # the last task in every chain starting on 'source'
            chain_ends = [chain[-1] for chain in chains]

            # group by chain name and account for all chunk blocks
            for _, subiter in groupby(chain_ends, itemgetter(0)):
                blocks = [item[1] for item in subiter]
                if not all(block in blocks for block in range(arr.numblocks[0])):
                    raise DaskGraphOptimizationFailure("missing blocks from calculation")

            # the input/output names for slicing tasks
            inname =  chain_ends[0][0] # input task name
            outname = 'selection-'+tokenize(arr,index,source)

            # determine the slice graph
            slice_dsk = {}
            for i, end in enumerate(chain_ends):
                new_key = list(end)
                new_key[0] = outname

                # this block number of this task
                blocknum = new_key[1]
                block_size = slice(chunk_slices[blocknum], chunk_slices[blocknum+1])
                index_this_block = index[block_size]

                # add the new slice
                slice_dsk[tuple(new_key)] = (getitem, end, numpy.where(index_this_block))

                # keep track of the size of this block
                blocksizes[blocknum] += index_this_block.sum()

            # skip this source if we didnt find the right size
            if sum(blocksizes.values()) != new_size:
               continue

            # if last getitem task is array name, we need to rename array
            if inname == arr.name:
                name = outname

            # update dependents of last consecutive getitem task
            # to point to "selection-*" tasks
            for k,v in slice_dsk.items():
                old_task_key = v[1]
                for dep in dependents[old_task_key]:

                    # try to get old task from new dask graph (dsk2) first
                    # in case we've already updated the task with selection tasks
                    old_task = dsk2.get(dep, dsk[dep])

                    # perform the substitution
                    dsk2[dep] = subs(old_task, old_task_key, k)

                # add the slice tasks to the new graph
                dsk2.update(slice_dsk)

    # if no new tasks, then we failed
    if not len(dsk2):
        args = (size, arr.catalog.size)
        raise DaskGraphOptimizationFailure("computed array length %d, not %d" % args)

    # new size has to be right!
    if sum(blocksizes.values()) != new_size:
        args = (sum(blocksizes.values()), new_size)
        raise DaskGraphOptimizationFailure("computed new sliced size %d, not %d" % args)

    # make the chunks
    chunks = ()
    for i in range(arr.numblocks[0]):
        chunks += (blocksizes[i],)
    chunks = (chunks,)
    chunks += arr.chunks[len(chunks):]

    # update the original graph and make new Array
    dsk.update(dsk2)
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

                if isinstance(sel, list):
                    sel = numpy.asarray(sel)
                if isinstance(sel, da.Array):
                    sel = self.catalog.compute(sel)

                # return self if the size is unchanged
                if sel.sum() == len(self):
                    return self

                # try to do the optimized selection
                try:
                    d = optimized_selection(self, sel)
                except DaskGraphOptimizationFailure as e:
                    if self.catalog.comm.rank == 0:
                        logging.debug("DaskGraphOptimizationFailure: %s" %str(e))
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
            r = r + " first: %s" % str(self[0].compute())
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
    An abstract base class that implements most of the functionality in
    :class:`CatalogSource`.

    The main difference between this class and :class:`CatalogSource` is that
    this base class does not assume the object has a :attr:`size` attribute.

    .. note::
        See the docstring for :class:`CatalogSource`. Most often, users should
        implement custom sources as subclasses of :class:`CatalogSource`.

    Parameters
    ----------
    comm :
        the MPI communicator to use for this object
    use_cache : bool, optional
        whether to cache intermediate dask task results; default is ``False``
    """
    logger = logging.getLogger('CatalogSourceBase')

    @staticmethod
    def make_column(array):
        """
        Utility function to convert an array-like object to a
        :class:`dask.array.Array`.

        .. note::
            The dask array chunk size is controlled via the ``dask_chunk_size``
            global option. See :class:`~nbodykit.set_options`.

        Parameters
        ----------
        array : array_like
            an array-like object; can be a dask array, numpy array,
            ColumnAccessor, or other non-scalar array-like object

        Returns
        -------
        :class:`dask.array.Array` :
            a dask array initialized from ``array``
        """
        if isinstance(array, da.Array):
            return array
        elif isinstance(array, ColumnAccessor):
            # important to get the accessor as a dask array to avoid circular
            # references
            return array.as_daskarray()
        else:
            return da.from_array(array, chunks=_global_options['dask_chunk_size'])

    def __new__(cls, *args, **kwargs):

        obj = object.__new__(cls)

        # ensure self.comm is set, though usually already set by the child.
        obj.comm = kwargs.get('comm', CurrentMPIComm.get())

        # initialize a cache
        obj.use_cache = kwargs.get('use_cache', False)

        # user-provided overrides for columns
        obj._overrides = {}

        # stores memory owner
        obj.base = None

        return obj

    def __finalize__(self, other):
        """
        Finalize the creation of a CatalogSource object by copying over
        any additional attributes from a second CatalogSource.

        The idea here is to only copy over attributes that are similar
        to meta-data, so we do not copy some of the core attributes of the
        :class:`CatalogSource` object.

        Parameters
        ----------
        other :
            the second object to copy over attributes from; it needs to be
            a subclass of CatalogSourcBase for attributes to be copied

        Returns
        -------
        CatalogSource :
            return ``self``, with the added attributes
        """
        if isinstance(other, CatalogSourceBase):
            d = other.__dict__.copy()
            nocopy = ['base', '_overrides', 'comm', '_cache', '_use_cache', '_size', '_csize']
            for key in d:
                if key not in nocopy:
                    self.__dict__[key] = d[key]

        return self

    def __iter__(self):
        return iter(self.columns)

    def __contains__(self, col):
        return col in self.columns

    def __slice__(self, index):
        """
        Select a subset of ``self`` according to a boolean index array.

        Returns a new object of the same type as ``selff`` holding only the
        data that satisfies the slice index.

        Parameters
        ----------
        index : array_like
            either a dask or numpy boolean array; this determines which
            rows are included in the returned object
        """
        # compute the index slice if needed and get the size
        if isinstance(index, da.Array):
            index = self.compute(index)
        elif isinstance(index, list):
            index = numpy.array(index)

        if getattr(self, 'size', NotImplemented) is NotImplemented:
            raise ValueError("cannot make catalog subset; self catalog doest not have a size")

        # verify the index is a boolean array
        if len(index) != self.size:
            raise ValueError("slice index has length %d; should be %d" %(len(index), self.size))
        if getattr(index, 'dtype', None) != '?':
            raise ValueError("index used to slice CatalogSource must be boolean and array-like")

        # new size is just number of True entries
        size = index.sum()

        # if size is the same, just return self
        if size == self.size:
           return self.base if self.base is not None else self

        # initialize subset Source of right size
        subset_data = {col:self[col][index] for col in self}
        cls = self.__class__ if self.base is None else self.base.__class__
        toret = cls._from_columns(size, self.comm, use_cache=self.use_cache, **subset_data)

        # attach the needed attributes
        toret.__finalize__(self)

        return toret

    def __getitem__(self, sel):
        """
        The following types of indexing are supported:

        #.  strings specifying a column in the CatalogSource; returns
            a dask array holding the column data
        #.  boolean arrays specifying a slice of the CatalogSource;
            returns a CatalogSource holding only the revelant slice
        #.  slice object specifying which particles to select
        #.  list of strings specifying column names; returns a CatalogSource
            holding only the selected columns

        .. note::
            If the :attr:`base` attribute is set, columns will be returned
            from :attr:`base` instead of from ``self``.
        """
        # handle boolean array slices
        if not isinstance(sel, string_types):

            # size must be implemented
            if getattr(self, 'size', NotImplemented) is NotImplemented:
                raise ValueError("cannot perform selection due to NotImplemented size")

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

                # convert into slice into boolean array
                index = numpy.zeros(self.size, dtype='?')
                index[sel] = True; sel = index

            # do the slicing
            if not numpy.isscalar(sel):
                return self.__slice__(sel)
            else:
                raise KeyError("strings and boolean arrays are the only supported indexing methods")

        # owner of the memory (either self or base)
        memowner = self if self.base is None else self.base

        # get the right column
        if sel in memowner._overrides:
            r = memowner._overrides[sel]
        elif sel in memowner.hardcolumns:
            r = memowner.get_hardcolumn(sel)
        else:
            raise KeyError("column `%s` is not defined in this source; " %sel + \
                            "try adding column via `source[column] = data`")

        # evaluate callables if we need to
        if callable(r): r = r()

        # return a ColumnAccessor for pretty prints
        return ColumnAccessor(memowner, r)

    def __setitem__(self, col, value):
        """
        Add new columns to the CatalogSource, overriding any existing columns
        with the name ``col``.

        .. note::
            If the :attr:`base` attribute is set, columns will be added to
            :attr:`base` instead of to ``self``.
        """
        if self.base is not None: return self.base.__setitem__(col, value)

        self._overrides[col] = self.make_column(value)

    def __delitem__(self, col):
        """
        Delete a column; cannot delete a "hard-coded" column.

        .. note::
            If the :attr:`base` attribute is set, columns will be deleted
            from :attr:`base` instead of from ``self``.
        """
        if self.base is not None: return self.base.__delitem__(col)

        if col not in self.columns:
            raise ValueError("no such column, cannot delete it")

        if col in self.hardcolumns:
            raise ValueError("cannot delete a hard-coded column")

        if col in self._overrides:
            del self._overrides[col]
            return

        raise ValueError("unable to delete column '%s' from CatalogSource" %col)

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

        See :class:`~nbodykit.set_options` to control the value of
        ``dask_cache_size``.
        """
        if val:
            try:
                from dask.cache import Cache
                if not hasattr(self, '_cache'):
                    self._cache = Cache(_global_options['dask_cache_size'])
            except ImportError:
                warnings.warn("caching of CatalogSource requires ``cachey`` module; turning cache off")
        else:
            if hasattr(self, '_cache'): delattr(self, '_cache')
        self._use_cache = val

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

    @property
    def hardcolumns(self):
        """
        A list of the hard-coded columns in the CatalogSource.

        These columns are usually member functions marked by ``@column``
        decorator. Subclasses may override this method and use
        :func:`get_hardcolumn` to bypass the decorator logic.

        .. note::
            If the :attr:`base` attribute is set, the value of
            ``base.hardcolumns`` will be returned.
        """
        if self.base is not None: return self.base.hardcolumns

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

        .. note::
            If the :attr:`base` attribute is set, the value of
            ``base.columns`` will be returned.
        """
        if self.base is not None: return self.base.columns

        hcolumns  = list(self.hardcolumns)
        overrides = list(self._overrides)
        return sorted(set(hcolumns + overrides))

    def copy(self):
        """
        Return a shallow copy of the object, where each column is a reference
        of the corresponding column in ``self``.

        .. note::
            No copy of data is made.

        Returns
        -------
        CatalogSource :
            a new CatalogSource that holds all of the data columns of ``self``
        """
        # a new empty object with proper size
        toret = CatalogSourceBase.__new__(self.__class__, self.comm, self.use_cache)
        toret._size = self.size
        toret._csize = self.csize

        # attach attributes from self and return
        toret = toret.__finalize__(self)

        # finally, add the data columns from self
        for col in self.columns:
            toret[col] = self[col]

        return toret

    def get_hardcolumn(self, col):
        """
        Construct and return a hard-coded column.

        These are usually produced by calling member functions marked by the
        ``@column`` decorator.

        Subclasses may override this method and the hardcolumns attribute to
        bypass the decorator logic.

        .. note::
            If the :attr:`base` attribute is set, ``get_hardcolumn()``
            will called using :attr:`base` instead of ``self``.
        """
        if self.base is not None: return self.base.get_hardcolumn(col)

        return find_column(self.__class__, col)(self)

    def compute(self, *args, **kwargs):
        """
        Our version of :func:`dask.compute` that computes
        multiple delayed dask collections at once.

        This should be called on the return value of :func:`read`
        to converts any dask arrays to numpy arrays.

        If :attr:`use_cache` is ``True``, this internally caches data, using
        dask's built-in cache features.

        .. note::
            If the :attr:`base` attribute is set, ``compute()``
            will called using :attr:`base` instead of ``self``.

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
        if self.base is not None: return self.base.compute(*args, **kwargs)

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

    def view(self, type=None):
        """
        Return a "view" of the CatalogSource object, with the returned
        type set by ``type``.

        This initializes a new empty class of type ``type`` and attaches
        attributes to it via the :func:`__finalize__` mechanism.

        Parameters
        ----------
        type : Python type
            the desired class type of the returned object.
        """
        # an empty class
        type = self.__class__ if type is None else type
        obj = CatalogSourceBase.__new__(type, self.comm, self.use_cache)

        # propagate the size attributes
        obj._size = self.size
        obj._csize = self.csize

        # the new object's base points to self
        obj.base = self

        # attach the necessary attributes from self
        return obj.__finalize__(self)

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

        return CatalogMesh(self, Nmesh=Nmesh,
                                 BoxSize=BoxSize,
                                 dtype=dtype,
                                 weight=weight,
                                 selection=selection,
                                 value=value,
                                 position=position,
                                 interlaced=interlaced,
                                 compensated=compensated,
                                 window=window)

class CatalogSource(CatalogSourceBase):
    """
    An abstract base class representing a catalog of discrete particles.

    This objects behaves like a structured numpy array -- it must have a
    well-defined size when initialized. The ``size`` here represents the
    number of particles in the source on the local rank.

    The information about each particle is stored as a series of
    columns in the format of dask arrays. These columns can be accessed
    in a dict-like fashion.

    All subclasses of this class contain the following default columns:

    #. ``Weight``
    #. ``Value``
    #. ``Selection``

    For a full description of these default columns, see
    :ref:`the documentation <catalog-source-default-columns>`.

    .. important::
        Subclasses of this class must set the ``_size`` attribute.

    Parameters
    ----------
    comm :
        the MPI communicator to use for this object
    use_cache : bool, optional
        whether to cache intermediate dask task results; default is ``False``
    """
    logger = logging.getLogger('CatalogSource')

    @classmethod
    def _from_columns(cls, size, comm, use_cache=False, **columns):
        """
        An internal constructor to create a CatalogSource (or subclass)
        from a set of columns.

        This method is used internally by nbodykit to create
        views of catalogs based on existing catalogs.

        Use :class:`~nbodykit.source.catalog.array.ArrayCatalog` to adapt
        a structured array or dictionary to a CatalogSource.

        .. note::
            The returned object is of type ``cls`` and the only attributes
            set for the returned object are :attr:`size` and :attr:`csize`.
        """
        # the new empty object to return
        obj = CatalogSourceBase.__new__(cls, comm, use_cache)

        # compute the sizes
        obj._size = size
        obj._csize = obj.comm.allreduce(obj._size)

        # add the columns in
        for name in columns:
            obj[name] = columns[name]

        return obj

    def __init__(self, *args, **kwargs):

        # if size is implemented, compute the csize
        if self.size is not NotImplemented:
            self._csize = self.comm.allreduce(self.size)
        else:
            self._csize = NotImplemented

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
            value = ConstantArray(value, self.size, chunks=_global_options['dask_chunk_size'])

        # check the correct size, if we know the size
        if self.size is not NotImplemented:
            args = (col, self.size, len(value))
            msg = "error setting '%s' column, data must be array of size %d, not %d" % args
            assert len(value) == self.size, msg

        # call the base __setitem__
        CatalogSourceBase.__setitem__(self, col, value)

    @property
    def size(self):
        """
        The number of objects in the CatalogSource on the local rank.

        If the :attr:`base` attribute is set, the ``base.size`` attribute
        will be returned.

        .. important::
            This property must be defined for all subclasses.
        """
        if self.base is not None: return self.base.size

        if not hasattr(self, '_size'):
            return NotImplemented
        return self._size

    @size.setter
    def size(self, value):
        raise RuntimeError(("Property size is read-only. Internally, ``_size`` "
                            "can be set during initialization."))

    @property
    def csize(self):
        """
        The total, collective size of the CatalogSource, i.e., summed across
        all ranks.

        It is the sum of :attr:`size` across all available ranks.

        If the :attr:`base` attribute is set, the ``base.csize`` attribute
        will be returned.
        """
        if self.base is not None: return self.base.csize

        if not hasattr(self, '_csize'):
            return NotImplemented
        return self._csize

    @column
    def Selection(self):
        """
        A boolean column that selects a subset slice of the CatalogSource.

        By default, this column is set to ``True`` for all particles.
        """
        return ConstantArray(True, self.size, chunks=_global_options['dask_chunk_size'])

    @column
    def Weight(self):
        """
        The column giving the weight to use for each particle on the mesh.

        The mesh field is a weighted average of ``Value``, with the weights
        given by ``Weight``.

        By default, this array is set to unity for all particles.
        """
        return ConstantArray(1.0, self.size, chunks=_global_options['dask_chunk_size'])

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
        return ConstantArray(1.0, self.size, chunks=_global_options['dask_chunk_size'])
