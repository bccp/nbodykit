from nbodykit.transform import ConstantArray
from nbodykit import _global_options, CurrentMPIComm

from six import string_types, add_metaclass
import numpy
import logging
import warnings
import abc
import inspect
import dask.array as da

class ColumnAccessor(da.Array):
    """
    Provides access to a Column from a Catalog.

    This is a thin subclass of :class:`dask.array.Array` to
    provide a reference to the catalog object,
    an additional ``attrs`` attribute (for recording the
    reproducible meta-data), and some pretty print support.

    Due to particularity of :mod:`dask`, any transformation
    that is not explicitly in-place will return
    a :class:`dask.array.Array`, and losing the pointer to
    the original catalog and the meta data attrs.

    Parameters
    ----------
    catalog : CatalogSource
        the catalog from which the column was accessed
    daskarray : dask.array.Array
        the column in dask array form
    is_default : bool, optional
        whether this column is a default column; default columns are not
        serialized to disk, as they are automatically available as columns

    """
    def __new__(cls, catalog, daskarray, is_default=False):
        self = da.Array.__new__(ColumnAccessor,
                dask=daskarray.dask,
                name=daskarray.name,
                chunks=daskarray.chunks,
                dtype=daskarray.dtype,
                shape=daskarray.shape)
        self.catalog = catalog
        self.is_default = is_default
        self.attrs = {}
        return self

    def __getitem__(self, key):

        # compute dask index b/c they are not fully supported
        if isinstance(key, da.Array):
            key = self.catalog.compute(key)

        # base class behavior
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

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        """
        Optimize the dask object.

        .. note::

            The dask default optimizer induces too many (unnecesarry)
            IO calls. We turn this feature off by default, and only apply a culling.

        """
        from dask.optimization import cull
        dsk2, dependencies = cull(dsk, keys)
        return dsk2

    def compute(self):
        return self.catalog.compute(self)

    def __str__(self):
        r = da.Array.__str__(self)
        if len(self) > 0:
            r = r + " first: %s" % str(self[0].compute())
        if len(self) > 1:
            r = r + " last: %s" % str(self[-1].compute())
        return r

def column(name=None, is_default=False):
    """
    Decorator that defines the decorated function as a column in a
    CatalogSource.

    This can be used as a decorator with or without arguments. If no ``name``
    is specified, the function name is used.

    Parameters
    ----------
    name : str, optional
        the name of the column; if not provided, the name of the function
        being decorated is used
    is_default : bool, optional
        whether the column is a default column; default columns are not
        serialized to disk
    """
    def decorator(getter, name=name):
        getter.column_name = getter.__name__ if name is None else name
        getter.is_default = is_default
        return getter

    # handle the case when decorator was called without arguments
    # in that case "name" is actually the function we are decorating
    if hasattr(name, '__call__'):
        getter = name
        return decorator(getter, name=getter.__name__)
    else:
        return decorator

class ColumnFinder(abc.ABCMeta):
    """
    A meta-class that will register all columns of a class that have
    been marked with the :func:`column` decorator.

    This adds the following attributes to the class definition:

    1. ``_defaults`` : default columns, specified by passing ``default=True`` to
    the :func:`column` decorator.

    2. ``_hardcolumns`` : non-default, hard-coded columns

    .. note::

        This is a subclass of :class:`abc.ABCMeta` so subclasses can
        define abstract properties, if they need to.

    """
    def __init__(cls, clsname, bases, attrs):

        # attach the registry attributes
        cls._defaults = set()
        cls._hardcolumns = set()

        # loop over class and its bases
        classes = inspect.getmro(cls)
        for c in reversed(classes):

            # loop over each attribute
            for name in c.__dict__:
                value = c.__dict__[name]

                # if it's member function implementing a column,
                # record it and check if its a default
                if getattr(value, 'column_name', None):
                    if value.is_default:
                        cls._defaults.add(value.column_name)
                    else:
                        cls._hardcolumns.add(value.column_name)

@add_metaclass(ColumnFinder)
class CatalogSourceBase(object):
    """
    An abstract base class that implements most of the functionality in
    :class:`CatalogSource`.

    The main difference between this class and :class:`CatalogSource` is that
    this base class does not assume the object has a :attr:`size` attribute.

    .. note::
        See the docstring for :class:`CatalogSource`. Most often, users should
        implement custom sources as subclasses of :class:`CatalogSource`.

    The names of hard-coded columns, i.e., those defined through member
    functions of the class, are stored in the :attr:`_defaults` and
    :attr:`_hardcolumns` attributes. These attributes are computed by the
    :class:`ColumnFinder` meta-class.

    Parameters
    ----------
    comm :
        the MPI communicator to use for this object
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

    @staticmethod
    def create_instance(cls, comm):

        obj = object.__new__(cls)
        CatalogSourceBase.__init__(obj, comm)

        return obj

    def __init__(self, comm):
        # user-provided overrides and defaults for columns
        self._overrides = {}

        # stores memory owner
        self.base = None

        self.comm = comm

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
            nocopy = ['base', '_overrides', '_hardcolumns', '_defaults', 'comm',
                      '_size', '_csize']
            for key in d:
                if key not in nocopy:
                    self.__dict__[key] = d[key]

        return self

    def __iter__(self):
        return iter(self.columns)

    def __contains__(self, col):
        return col in self.columns

    def _get_slice(self, index):
        """
        Select a subset of ``self`` according to a boolean index array.

        Returns a new object of the same type as ``self`` holding only the
        data that satisfies the slice index.

        Parameters
        ----------
        index : array_like
            either a dask or numpy boolean array; this determines which
            rows are included in the returned object
        """
        if index is Ellipsis:
           return self
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.size)
            # from https://stackoverflow.com/a/36188683
            size = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
        else:
            # compute the index slice if needed and get the size
            index = CatalogSourceBase.make_column(index)

            if index.dtype == numpy.dtype('?'):
                # verify the index is a boolean array
                if len(index) != self.size:
                    raise KeyError("slice index has length %d; should be %d" %(len(index), self.size))

                # new size is just number of True entries
                size = index.sum().compute()
            else:

                if len(index) > 0 and not numpy.issubdtype(index.dtype, numpy.integer):
                    raise KeyError("slice index has must be boolean, integer. got %s" %(index.dtype))

                size = len(index)

        # initialize subset Source of right size
        subset_data = {col:self[col][index] for col in self if not self[col].is_default}
        if size <= 0.51 * self.size:
            # if the subsample ratio is substential, then always make
            # a copy to decouple from the original data
            subset_data = {col:subset_data[col].map_blocks(numpy.copy) for col in subset_data}

        cls = self.__class__ if self.base is None else self.base.__class__
        toret = cls._from_columns(size, self.comm, **subset_data)

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

        Notes
        -----
        - Slicing is a **collective** operation
        - If the :attr:`base` attribute is set, columns will be returned
          from :attr:`base` instead of from ``self``.
        """
        # handle boolean array slices
        if not isinstance(sel, string_types):

            # size must be implemented
            if getattr(self, 'size', NotImplemented) is NotImplemented:
                raise ValueError("cannot perform selection due to NotImplemented size")

            # convert slices to boolean arrays
            if isinstance(sel, (list, da.Array, numpy.ndarray)):

                # select a subset of list of string column names
                if len(sel) > 0 and all(isinstance(ss, string_types) for ss in sel):
                    invalid = set(sel) - set(self.columns)
                    if len(invalid):
                        msg = "cannot select subset of columns from "
                        msg += "CatalogSource due to invalid columns: %s" % str(invalid)
                        raise KeyError(msg)

                    # return a CatalogSource only holding the selected columns
                    subset_data = {col:self[col] for col in sel}
                    toret = CatalogSource._from_columns(self.size, self.comm, **subset_data)
                    toret.attrs.update(self.attrs)
                    return toret

            return self._get_slice(sel)
        else:
            # owner of the memory (either self or base)
            if self.base is None:
                # get the right column
                is_default = False
                if sel in self._overrides:
                    r = self._overrides[sel]
                elif sel in self.hardcolumns:
                    r = self.get_hardcolumn(sel)
                elif sel in self._defaults:
                    r = getattr(self, sel)()
                    is_default = True
                else:
                    raise KeyError("column `%s` is not defined in this source; " %sel + \
                                    "try adding column via `source[column] = data`")
                # return a ColumnAccessor for pretty prints
                return ColumnAccessor(self, r, is_default=is_default)
            else:
                # chain to the memory owner
                # this will not work if there are overrides
                return self.base.__getitem__(sel)


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

        # return the non-default, hard-coded columns, as determined by
        # ColumnFinder metaclass
        return sorted(self._hardcolumns)

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

        overrides = list(self._overrides)
        defaults = list(self._defaults)
        return sorted(set(self.hardcolumns + overrides + defaults))

    def copy(self):
        """
        Return a shallow copy of the object, where each column is a reference
        of the corresponding column in ``self``.

        .. note::
            No copy of data is made.

        .. note::
            This is different from view in that the attributes dictionary
            of the copy no longer related to ``self``.

        Returns
        -------
        CatalogSource :
            a new CatalogSource that holds all of the data columns of ``self``
        """
        # a new empty object with proper size
        toret = CatalogSourceBase.create_instance(self.__class__, comm=self.comm)
        toret._size = self.size
        toret._csize = self.csize

        # attach attributes from self and return
        toret = toret.__finalize__(self)

        # finally, add the data columns from self
        for col in self.columns:
            toret[col] = self[col]

        # copy the attributes too, so they become decoupled
        # this is different from view.
        toret._attrs = self._attrs.copy()

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

        if col in self._hardcolumns:
            return getattr(self, col)()
        else:
            raise ValueError("no such hard-coded column %s" %col)

    def compute(self, *args, **kwargs):
        """
        Our version of :func:`dask.compute` that computes
        multiple delayed dask collections at once.

        This should be called on the return value of :func:`read`
        to converts any dask arrays to numpy arrays.

        . note::
            If the :attr:`base` attribute is set, ``compute()``
            will called using :attr:`base` instead of ``self``.

        Parameters
        -----------
        args : object
            Any number of objects. If the object is a dask
            collection, it's computed and the result is returned.
            Otherwise it's passed through unchanged.

        """
        import dask

        # return the base compute if it exists
        if self.base is not None:
            return self.base.compute(*args, **kwargs)

        toret = dask.compute(*args, **kwargs)

        # do not return tuples of length one
        if len(toret) == 1: toret = toret[0]
        return toret

    def save(self, output, columns=None, dataset=None, datasets=None, header='Header', compute=True):
        """
        Save the CatalogSource to a :class:`bigfile.BigFile`.

        Only the selected columns are saved and :attr:`attrs` are saved in
        ``header``. The attrs of columns are stored in the datasets.

        Parameters
        ----------
        output : str
            the name of the file to write to
        columns : list of str
            the names of the columns to save in the file, or None to use all columns
        dataset : str, optional
            dataset to store the columns under.
        datasets : list of str, optional
            names for the data set where each column is stored; defaults to
            the name of the column (deprecated)
        header : str, optional, or None
            the name of the data set holding the header information, where
            :attr:`attrs` is stored
            if header is None, do not save the header.
        compute : boolean, default True
            if True, wait till the store operations finish
            if False, return a dictionary with column name and a future object for the store.
            use dask.compute() to wait for the store operations on the result.
        """
        import bigfile
        import json
        from nbodykit.utils import JSONEncoder

        # trim out any default columns; these do not need to be saved as
        # they are automatically available to every Catalog
        if columns is None: columns = self.columns
        columns = [col for col in columns if not self[col].is_default]

        # also make sure no default columns in datasets
        if datasets is not None:
            import warnings
            warnings.warn("datasets argument is deprecated. Specify a single dataset prefix for all columns instead.")
        else:
            if dataset is not None:
                datasets = [ dataset + '/' + col for col in columns ]
            else:
                datasets = columns

        if len(datasets) != len(columns):
            raise ValueError("`datasets` must have the same length as `columns`")

        # FIXME: merge this logic into bigfile
        # the slice writing support in bigfile 0.1.47 does not
        # support tuple indices.
        class _ColumnWrapper:
            def __init__(self, bb):
                self.bb = bb
            def __setitem__(self, sl, value):
                assert len(sl) <= 2 # no array shall be of higher dimension.
                # use regions argument to pick the offset.
                start, stop, step = sl[0].indices(self.bb.size)
                assert step == 1
                if len(sl) > 1:
                    start1, stop1, step1 = sl[1].indices(value.shape[1])
                    assert step1 == 1
                    assert start1 == 0
                    assert stop1 == value.shape[1]
                self.bb.write(start, value)

        with bigfile.FileMPI(comm=self.comm, filename=output, create=True) as ff:

            sources = []
            targets = []
            regions = []

            # save meta data and create blocks, prepare for the write.
            for column, dataset in zip(columns, datasets):
                array = self[column]
                # ensure data is only chunked in the first dimension
                size = self.comm.allreduce(len(array))
                offset = numpy.sum(self.comm.allgather(len(array))[:self.comm.rank], dtype='i8')

                # sane value -- 32 million items per physical file
                sizeperfile = 32 * 1024 * 1024

                Nfile = (size + sizeperfile - 1) // sizeperfile

                dtype = numpy.dtype((array.dtype, array.shape[1:]))

                # save column attrs too
                # first create the block on disk
                with ff.create(dataset, dtype, size, Nfile) as bb:
                    if hasattr(array, 'attrs'):
                        for key in array.attrs:
                            bb.attrs[key] = array.attrs[key]

                # first then open it for writing
                bb = ff.open(dataset)

                # ensure only the first dimension is chunked
                # because bigfile only support writing with slices in first dimension.
                rechunk = dict([(ind, -1) for ind in range(1, array.ndim)])
                array = array.rechunk(rechunk)

                targets.append(_ColumnWrapper(bb))
                sources.append(array)
                regions.append((slice(offset, offset + len(array)),))

            # writer header afterwards, such that header can be a block that saves
            # data.
            if header is not None:
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

            # lock=False to avoid dask from pickling the lock with the object.
            if compute:
                # write blocks one by one
                for column, source, target, region in zip(columns, sources, targets, regions):
                    if self.comm.rank == 0:
                        self.logger.info("started writing column %s" % column)
                    source.store(target, regions=region, lock=False, compute=True)
                    target.bb.close()
                    if self.comm.rank == 0:
                        self.logger.info("finished writing column %s" % column)
                future = None
            else:
                # return a future that writes all blocks at the same time.
                # Note that must pass in lists, not tuples or da.store is confused.
                # c.f https://github.com/dask/dask/issues/4393
                future = da.store(sources, targets, regions=regions, lock=False, compute=False)

        return future

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
        obj = CatalogSourceBase.create_instance(type, comm=self.comm)

        # propagate the size attributes
        obj._size = self.size
        obj._csize = self.csize

        # the new object's base points to self
        obj.base = self

        # attach the necessary attributes from self
        return obj.__finalize__(self)

    def to_subvolumes(self, domain=None, position='Position', columns=None):
        """
        Domain Decompose a catalog, sending items to the ranks according to the
        supplied domain object. Using the `position` column as the Position.

        This will read in the full position array and all of the requested columns.

        Parameters
        ----------
        domain : :class:`pmesh.domain.GridND` object, or None
            The domain to distribute the catalog. If None, try to evenly divide
            spatially.
            An easiest way to find a domain object is to use `pm.domain`, where `pm`
            is a :class:`pmesh.pm.ParticleMesh` object.

        position : string_like
            column to use to compute the position.

        columns: list of string_like
            columns to include in the new catalog, if not supplied, all catalogs
            will be exchanged.

        Returns
        -------
        CatalogSource
            A decomposed catalog source, where each rank only contains objects
            belongs to the rank as claimed by the domain object.

            `self.attrs` are carried over as a shallow copy to the returned object.
        """
        from nbodykit.source.catalog import SubVolumesCatalog
        return SubVolumesCatalog(self, domain=domain, position=position, columns=columns)

    def to_mesh(self, Nmesh=None, BoxSize=None, dtype='f4', interlaced=False,
                compensated=False, resampler='cic', weight='Weight',
                value='Value', selection='Selection', position='Position', window=None):
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
            whether to correct for the resampler window introduced by the grid
            interpolation scheme
        resampler : str, optional
            the string specifying which resampler interpolation scheme to use;
            see `pmesh.resampler.methods`
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
        window : str, deprecated
            use resampler instead.

        Returns
        -------
        mesh : CatalogMesh
            a mesh object that provides an interface for gridding particle
            data onto a specified mesh
        """
        from nbodykit.source.mesh import CatalogMesh
        from pmesh.window import methods

        if window is not None:
            resampler = window
            import warnings
            warnings.warn("The window argument is deprecated. Use `resampler=` instead", DeprecationWarning, stacklevel=2)

        # make sure all of the columns exist
        for col in [weight, selection]:
            if col not in self:
                raise ValueError("column '%s' missing; cannot create mesh" %col)

        if resampler not in methods:
            raise ValueError("valid resampler: %s" %str(methods))

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
                                 Weight=self[weight],
                                 Selection=self[selection],
                                 Value=self[value],
                                 Position=self[position],
                                 interlaced=interlaced,
                                 compensated=compensated,
                                 resampler=resampler)

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
    """
    logger = logging.getLogger('CatalogSource')

    @classmethod
    def _from_columns(cls, size, comm, **columns):
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
        obj = CatalogSourceBase.create_instance(cls, comm=comm)

        # compute the sizes
        obj._size = size
        obj._csize = obj.comm.allreduce(obj._size)

        # add the columns in
        for name in columns:
            obj[name] = columns[name]

        return obj

    def __init__(self, comm):

        CatalogSourceBase.__init__(self, comm)

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

    def gslice(self, start, stop, end=1, redistribute=True):
        """
        Execute a global slice of a CatalogSource.

        .. note::

            After the global slice is performed, the data is scattered
            evenly across all ranks.

        .. note::

            The current algorithm generates an index on the root rank
            and does not scale well.

        Parameters
        ----------
        start : int
            the start index of the global slice
        stop : int
            the stop index of the global slice
        step : int, optional
            the default step size of the global size
        redistribute : bool, optional
            if ``True``, evenly re-distribute the sliced data across all
            ranks, otherwise just return any local data part of the global
            slice
        """
        from nbodykit.utils import ScatterArray, GatherArray

        # determine the boolean index corresponding to the slice
        if self.comm.rank == 0:
            index = numpy.zeros(self.csize, dtype=bool)
            index[slice(start, stop, end)] = True
        else:
            index = None
        index = self.comm.bcast(index)

        # scatter the index back to all ranks
        counts = self.comm.allgather(self.size)
        index = ScatterArray(index, self.comm, root=0, counts=counts)

        # perform the needed local slice
        subset = self[index]

        # if we don't want to redistribute evenly, just return the slice
        if not redistribute:
            return subset

        # re-distribute each column from the sliced data
        # NOTE: currently Gather/ScatterArray requires numpy arrays, but
        # in principle we could pass dask arrays around between ranks and
        # avoid compute() calls
        data = self.compute(*[subset[col] for col in subset])

        # gather/scatter each column
        evendata = {}
        for i, col in enumerate(subset):
            alldata = GatherArray(data[i], self.comm, root=0)
            evendata[col] = ScatterArray(alldata, self.comm, root=0)

        # return a new CatalogSource holding the evenly distributed data
        size = len(evendata[col])
        toret = self.__class__._from_columns(size, self.comm, **evendata)
        return toret.__finalize__(self)

    def persist(self, columns=None):
        """
        Return a CatalogSource, where the selected columns are
        computed and persist in memory.
        """

        import dask.array as da
        if columns is None:
            columns = self.columns

        r = {}
        for key in columns:
            r[key] = self[key]

        r = da.compute(r)[0] # particularity of dask

        from nbodykit.source.catalog.array import ArrayCatalog
        c = ArrayCatalog(r, comm=self.comm)
        c.attrs.update(self.attrs)

        return c

    def sort(self, keys, reverse=False, usecols=None):
        """
        Return a CatalogSource, sorted globally across all MPI ranks
        in ascending order by the input keys.

        Sort columns must be floating or integer type.

        .. note::
            After the sort operation, the data is scattered evenly across
            all ranks.

        Parameters
        ----------
        keys : list, tuple
            the names of columns to sort by. If multiple columns are provided,
            the data is sorted consecutively in the order provided
        reverse : bool, optional
            if ``True``, perform descending sort operations
        usecols : list, optional
            the name of the columns to include in the returned CatalogSource
        """
        # single string passed as input
        if isinstance(keys, string_types):
            keys = [keys]

        # no duplicated keys
        if len(set(keys)) != len(keys):
            raise ValueError("duplicated sort keys")

        # all keys must be valid
        bad = set(keys) - set(self.columns)
        if len(bad):
            raise ValueError("invalid sort keys: %s" %str(bad))

        # check usecols input
        if usecols is not None:
            if isinstance(usecols, string_types):
                usecols = [usecols]
            if not isinstance(usecols, (list, tuple)):
                raise ValueError("usecols should be a list or tuple of column names")

            bad = set(usecols) - set(self.columns)
            if len(bad):
                raise ValueError("invalid column names in usecols: %s" %str(bad))

        # sort the data
        data = _sort_data(self.comm, self, list(keys), reverse=reverse, usecols=usecols)

        # get a dictionary of data
        cols = {}
        for col in data.dtype.names:
            cols[col] = data[col]

        # make the new CatalogSource from the data dict
        size = len(data)

        # NOTE: if we are slicing by columns too, we must return a bare CatalogSource
        # we cannot guarantee the consistency of the hard column definitions otherwise
        if usecols is None:
            toret = self.__class__._from_columns(size, self.comm, **cols)
            return toret.__finalize__(self)
        else:
            toret = CatalogSource._from_columns(size, self.comm, **cols)
            toret.attrs.update(self.attrs)
            return toret

    @column(is_default=True)
    def Selection(self):
        """
        A boolean column that selects a subset slice of the CatalogSource.

        By default, this column is set to ``True`` for all particles, and
        all CatalogSource objects will contain this column.
        """
        return ConstantArray(True, self.size, chunks=_global_options['dask_chunk_size'])

    @column(is_default=True)
    def Weight(self):
        """
        The column giving the weight to use for each particle on the mesh.

        The mesh field is a weighted average of ``Value``, with the weights
        given by ``Weight``.

        By default, this array is set to unity for all particles, and
        all CatalogSource objects will contain this column.
        """
        return ConstantArray(1.0, self.size, chunks=_global_options['dask_chunk_size'])

    @property
    def Index(self):
        """
        The attribute giving the global index rank of each particle in the
        list. It is an integer from 0 to ``self.csize``.

        Note that slicing changes this index value.
        """
        offset = numpy.sum(self.comm.allgather(self.size)[:self.comm.rank], dtype='intp')
        # do not use u8, because many numpy casting rules case u8 to f8 automatically.
        # it is ridiculous.
        return da.arange(offset, offset + self.size, dtype='i8',
               chunks=_global_options['dask_chunk_size'])

    @column(is_default=True)
    def Value(self):
        """
        When interpolating a CatalogSource on to a mesh, the value of this
        array is used as the Value that each particle contributes to a given
        mesh cell.

        The mesh field is a weighted average of ``Value``, with the weights
        given by ``Weight``.

        By default, this array is set to unity for all particles, and
        all CatalogSource objects will contain this column.
        """
        return ConstantArray(1.0, self.size, chunks=_global_options['dask_chunk_size'])


def _sort_data(comm, cat, rankby, reverse=False, usecols=None):
    """
    Sort the input data by the specified columns

    Parameters
    ----------
    comm :
        the mpi communicator
    cat : CatalogSource
        the catalog holding the data to sort
    rankby : list of str
        list of columns to sort by
    reverse : bool, optional
        if ``True``, sort in descending order
    usecols : list, optional
        only sort these data columns
    """
    import mpsort

    # determine which columns we need
    if usecols is None:
        usecols = cat.columns

    # remove duplicates from usecols
    usecols = list(set(usecols))

    # the columns we need in the sort steps
    columns = list(set(rankby)|set(usecols))

    # make the data to sort
    dtype = [('_sortkey', 'i8')]
    for col in cat:
        if col in columns:
            dt = (cat[col].dtype.char,)
            dt += cat[col].shape[1:]
            if len(dt) == 1: dt = dt[0]
            dtype.append((col, dt))
    dtype = numpy.dtype(dtype)

    data = numpy.empty(cat.size, dtype=dtype)
    for col in columns:
        data[col] = cat[col]

    # sort the particles by the specified columns and store the
    # corrected sorted index
    for col in reversed(rankby):
        dt = data.dtype[col]
        rankby_name = col

        # make an integer key for floating columns
        # this assumes the lexial order of float as integer is consistant.
        if issubclass(dt.type, numpy.float32):
            data['_sortkey'] = numpy.frombuffer(data[col].tobytes(), dtype='i4')
            if reverse:
                data['_sortkey'] *= -1
            rankby_name = '_sortkey'
        elif issubclass(dt.type, numpy.float64):
            data['_sortkey'] = numpy.frombuffer(data[col].tobytes(), dtype='i8')
            if reverse:
                data['_sortkey'] *= -1
            rankby_name = '_sortkey'
        elif not issubclass(dt.type, numpy.integer):
            args = (col, str(dt))
            raise ValueError("cannot sort by column '%s' with dtype '%s'; must be integer or floating type" %args)

        # do the parallel sort
        mpsort.sort(data, orderby=rankby_name, comm=comm)

    return data[usecols]
