from six import add_metaclass, string_types
from nbodykit.transform import ConstantArray

import abc
import numpy
import logging
import warnings
import dask.array as da

# default size of Cache for CatalogSource arrays
CACHE_SIZE = 1e9


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
        else:
            return da.from_array(array, chunks=100000)

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
            returns a CatalogCopy holding only the revelant slice
        #.  slice object specifying which particles to select
        #.  list of strings specifying column names; returns a CatalogCopy
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
                    toret = CatalogCopy(self.size, self.comm, use_cache=self.use_cache, **subset_data)
                    toret.attrs.update(self.attrs)
                    return toret

                # list must be all integers
                if isinstance(sel, list) and not numpy.array(sel).dtype == numpy.integer:
                    raise KeyError("array like indexing via a list should be a list of integers")

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

        # add a column attrs dict
        if not hasattr(r, 'attrs'): r.attrs = {}

        return r

    def __setitem__(self, col, value):
        """
        Add new columns to the CatalogSource, overriding any existing columns
        with the name ``col``.
        """
        self._overrides[col] = self.make_column(value)

    def __delitem__(self, col):
        """
        Delete a column; cannot delete a "hard-coded" column
        """
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
        Initialize a Cache object of size set by ``CACHE_SIZE``, which
        is 1 GB by default.
        """
        if val:
            try:
                from dask.cache import Cache
                if not hasattr(self, '_cache'):
                    self._cache = Cache(CACHE_SIZE)
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
            value = ConstantArray(value, self.size, chunks=100000)

        # check the correct size, if we know the size
        if self.size is not NotImplemented:
            assert len(value) == self.size, "error setting column, data must be array of size %d" %self.size

        # call the base __setitem__
        CatalogSourceBase.__setitem__(self, col, value)

    def copy(self):
        """
        Return a copy of the CatalogSource object

        Returns
        -------
        CatalogCopy :
            the new CatalogSource object holding the copied data columns
        """
        if self.size is NotImplemented:
            return ValueError("cannot copy a CatalogSource that does not have `size` implemented")

        data = {col:self[col] for col in self.columns}
        toret = CatalogCopy(self.size, comm=self.comm, use_cache=self.use_cache, **data)
        toret.attrs.update(self.attrs)
        return toret

    @abc.abstractproperty
    def size(self):
        """
        The number of particles in the CatalogSource on the local rank.

        This property must be defined for all subclasses.
        """
        return NotImplemented

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
        return ConstantArray(True, self.size, chunks=100000)

    @column
    def Weight(self):
        """
        The column giving the weight to use for each particle on the mesh.

        The mesh field is a weighted average of ``Value``, with the weights
        given by ``Weight``.

        By default, this array is set to unity for all particles.
        """
        return ConstantArray(1.0, self.size, chunks=100000)

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
        return ConstantArray(1.0, self.size, chunks=100000)

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


class CatalogCopy(CatalogSource):
    """
    A CatalogSource object that holds column data copied from an
    original source

    Parameters
    ----------
    size : int
        the size of the new source; this was likely determined by
        the number of particles passing the selection criterion
    comm : MPI communicator
        the MPI communicator; this should be the same as the
        comm of the object that we are selecting from
    use_cache : bool, optional
        whether to cache results
    **columns :
        the data arrays that will be added to this source; keys
        represent the column names
    """
    def __init__(self, size, comm, use_cache=False, **columns):

        self._size = size
        CatalogSource.__init__(self, comm=comm, use_cache=use_cache)

        # store the column arrays
        for name in columns:
            self[name] = columns[name]

    @property
    def size(self):
        return self._size

def get_catalog_subset(parent, index):
    """
    Select a subset of a :class:`CatalogSource` according to a boolean
    index array.

    Returns a :class:`CatalogCopy` holding only the data that satisfies
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
    subset : :class:`CatalogCopy`
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
    toret = CatalogCopy(size, parent.comm, use_cache=parent.use_cache, **subset_data)

    # and the meta-data
    toret.attrs.update(parent.attrs)

    return toret
