from nbodykit.base.catalog import CatalogSource
from nbodykit.io.stack import FileStack
from nbodykit import CurrentMPIComm
from nbodykit import io
from nbodykit.extern import docrep

from six import string_types
import textwrap
import os

__all__ = ['FileCatalogFactory', 'FileCatalogBase',
           'CSVCatalog', 'BinaryCatalog', 'BigFileCatalog',
           'HDFCatalog', 'TPMBinaryCatalog', 'Gadget1Catalog', 'FITSCatalog']

class FileCatalogBase(CatalogSource):
    """
    Base class to create a source of particles from a
    single file, or multiple files, on disk.

    Files of a specific type should be subclasses of this class.

    Parameters
    ----------
    filetype : subclass of :class:`~nbodykit.io.base.FileType`
        the file-like class used to load the data from file; should be a
        subclass of :class:`nbodykit.io.base.FileType`
    args : tuple, optional
        the arguments to pass to the ``filetype`` class when constructing
        each file object
    kwargs : dict, optional
        the keyword arguments to pass to the ``filetype`` class when
        constructing each file object
    comm : MPI Communicator, optional
        the MPI communicator instance; default (``None``) sets to the
        current communicator
    """
    @CurrentMPIComm.enable
    def __init__(self, filetype, args=(), kwargs={}, comm=None):

        self.comm = comm
        self.filetype = filetype

        # bcast the FileStack
        if self.comm.rank == 0:
            self._source = FileStack(filetype, *args, **kwargs)
        else:
            self._source = None
        self._source = self.comm.bcast(self._source)

        # compute the size
        start = self.comm.rank * self._source.size // self.comm.size
        end = (self.comm.rank  + 1) * self._source.size // self.comm.size
        self._size = end - start

        # update the meta-data
        self.attrs.update(self._source.attrs)

        if self.comm.rank == 0:
            self.logger.info("Extra arguments to FileType: %s" % str(args))

        CatalogSource.__init__(self, comm=comm)

    def __repr__(self):
        path = self._source.path
        name = self.__class__.__name__
        if isinstance(path, string_types):
            args = (name, self.size, os.path.basename(path))
            return "%s(size=%d, file='%s')" % args
        else:
            args = (name, self.size, self._source.nfiles)
            return "%s(size=%d, nfiles=%d)" % args

    @property
    def hardcolumns(self):
        """
        The union of the columns in the file and any transformed columns.
        """
        defaults = CatalogSource.hardcolumns.fget(self)
        return list(self._source.dtype.names) + defaults

    def get_hardcolumn(self, col):
        """
        Return a column from the underlying file source.

        Columns are returned as dask arrays.
        """
        if col in self._source.dtype.names:
            start = self.comm.rank * self._source.size // self.comm.size
            end = (self.comm.rank  + 1) * self._source.size // self.comm.size
            return self._source.get_dask(col)[start:end]
        else:
            return CatalogSource.get_hardcolumn(self, col)


def _make_docstring(filetype, examples):
    """
    Internal function to generate the doc strings for the built-in
    CatalogSource objects that rely on :mod:`nbodykit.io` classes
    to read data from disk.
    """

    qualname = '%s.%s' %(filetype.__module__, filetype.__name__)
    __doc__ = """
A CatalogSource that uses :class:`~{qualname}` to read data from disk.

Multiple files can be read at once by supplying a list of file
names or a glob asterisk pattern as the ``path`` argument. See
:ref:`reading-multiple-files` for examples.

Parameters
----------
%(test.parameters)s
comm : MPI Communicator, optional
    the MPI communicator instance; default (``None``) sets to the
    current communicator
attrs : dict, optional
    dictionary of meta-data to store in :attr:`attrs`
""".format(qualname=qualname)

    if examples is not None:
        __doc__ += """
Examples
--------
Please see :ref:`the documentation <%s>` for examples.
""" %examples

    # get the Parameters from the IO libary class
    d = docrep.DocstringProcessor()
    d.get_sections(d.dedents(filetype.__doc__), 'test', ['Parameters'])
    return d.dedents(__doc__)

def FileCatalogFactory(name, filetype, examples=None):
    """
    Factory method to create a :class:`~nbodykit.base.catalog.CatalogSource`
    that uses a subclass of :mod:`nbodykit.io.base.FileType` to read
    data from disk.

    Parameters
    ----------
    name : str
        the name of the catalog class to create
    filetype : subclass of :class:`nbodykit.io.base.FileType`
        the subclass of the FileType that reads a specific type of data
    examples : str, optional
        if given, a documentation cross-reference link where examples can be
        found

    Returns
    -------
    subclass of :class:`FileCatalogBase` :
        the ``CatalogSource`` object that reads data using ``filetype``
    """
    def __init__(self, *args, **kwargs):
        comm = kwargs.pop('comm', None)
        attrs = kwargs.pop('attrs', {})
        FileCatalogBase.__init__(self, filetype=filetype, args=args, kwargs=kwargs, comm=comm)
        self.attrs.update(attrs)

    # make the doc string for this class
    __doc__ = _make_docstring(filetype, examples)

    # make the new class object and return it
    newclass = type(name, (FileCatalogBase,),{"__init__": __init__, "__doc__":__doc__})
    return newclass

CSVCatalog       = FileCatalogFactory("CSVCatalog", io.CSVFile, examples='csv-data')
BinaryCatalog    = FileCatalogFactory("BinaryCatalog", io.BinaryFile, examples='binary-data')
BigFileCatalog   = FileCatalogFactory("BigFileCatalog", io.BigFile, examples='bigfile-data')
HDFCatalog       = FileCatalogFactory("HDFCatalog", io.HDFFile, examples='hdf-data')
TPMBinaryCatalog = FileCatalogFactory("TPMBinaryCatalog", io.TPMBinaryFile)
FITSCatalog      = FileCatalogFactory("FITSCatalog", io.FITSFile, examples='fits-data')
Gadget1Catalog   = FileCatalogFactory("Gadget1Catalog", io.Gadget1File, examples=None)
