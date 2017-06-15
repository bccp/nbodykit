from nbodykit.base.catalog import CatalogSource
from nbodykit.io.stack import FileStack
from nbodykit import CurrentMPIComm
from nbodykit import io

class FileCatalogBase(CatalogSource):
    """
    Base class to create a source of particles from a
    single file, or multiple files, on disk

    Files of a specific type should be subclasses of this class.
    """
    @CurrentMPIComm.enable
    def __init__(self, filetype, args=(), kwargs={}, comm=None, use_cache=False):
        """
        Parameters
        ----------
        filetype : subclass of :class:`nbodykit.io.FileType`
            the file-like class used to load the data from file; should be a
            subclass of :class:`nbodykit.io.FileType`
        path : str, list of str
            the path to the file(s) to load; can be a list of files to load, or
            contain a glob-like asterisk pattern
        args : dict, optional
            the arguments to pass to the ``filetype`` class when constructing
            each file object
        """
        self.comm = comm
        self.filetype = filetype

        # bcast the FileStack
        if self.comm.rank == 0:
            self._source = FileStack(filetype, *args, **kwargs)
        else:
            self._source = None
        self._source = self.comm.bcast(self._source)

        # update the meta-data
        self.attrs.update(self._source.attrs)

        if self.comm.rank == 0:
            self.logger.info("Extra arguments to FileType: %s" % str(args))

        CatalogSource.__init__(self, comm=comm, use_cache=use_cache)

    @property
    def size(self):
        """
        The local size
        """
        start = self.comm.rank * self._source.size // self.comm.size
        end = (self.comm.rank  + 1) * self._source.size // self.comm.size
        return end - start

    @property
    def hardcolumns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        defaults = CatalogSource.hardcolumns.fget(self)
        return list(self._source.dtype.names) + defaults

    def get_hardcolumn(self, col):
        """
        Return a column from the underlying file source

        Columns are returned as dask arrays
        """
        if col in self._source.dtype.names:
            start = self.comm.rank * self._source.size // self.comm.size
            end = (self.comm.rank  + 1) * self._source.size // self.comm.size
            return self._source.get_dask(col)[start:end]
        else:
            return CatalogSource.get_hardcolumn(self, col)


def FileCatalogFactory(name, filetype):
    """
    Factory method to create subclasses of :class:`FileCatalogBase`
    that use specific classes from :mod:`nbodykit.io` to read
    different types of data from disk
    """
    def wrapdoc(cls):
        """Add the docstring of the IO class"""
        def dec(f):
            io_doc = cls.__init__.__doc__
            if io_doc is None: io_doc = ""
            f.__doc__ =  io_doc + f.__doc__
            return f
        return dec

    @wrapdoc(filetype)
    def __init__(self, *args, **kwargs):
        """
        Additional Keyword Parameters
        -----------------------------
        comm : MPI Communicator, optional
            the MPI communicator instance; default (``None``) sets to the
            current communicator
        use_cache : bool, optional
            whether to cache data read from disk; default is ``False``
        attrs : dict; optional
            dictionary of meta-data to store in :attr:`attrs`
        """
        comm = kwargs.pop('comm', None)
        use_cache = kwargs.pop('use_cache', False)
        attrs = kwargs.pop('attrs', {})
        FileCatalogBase.__init__(self, filetype=filetype, args=args, kwargs=kwargs)
        self.attrs.update(attrs)


    __doc__ = "A catalog source created using :class:`io.%s` to read data from disk" % filetype.__name__
    newclass = type(name, (FileCatalogBase,),{"__init__": __init__, "__doc__":__doc__})
    return newclass

CSVCatalog       = FileCatalogFactory("CSVCatalog", io.CSVFile)
BinaryCatalog    = FileCatalogFactory("BinaryCatalog", io.BinaryFile)
BigFileCatalog   = FileCatalogFactory("BigFileCatalog", io.BigFile)
HDFCatalog       = FileCatalogFactory("HDFCatalog", io.HDFFile)
TPMBinaryCatalog = FileCatalogFactory("TPMBinaryCatalog", io.TPMBinaryFile)
FITSCatalog      = FileCatalogFactory("FITSCatalog", io.FITSFile)
