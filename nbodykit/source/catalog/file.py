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
    path : string or list. If string, it is expanded as a glob pattern.
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
    def __init__(self, filetype, path, args=(), kwargs={}, comm=None):

        self.comm = comm
        self.filetype = filetype

        # bcast the FileStack
        if self.comm.rank == 0:
            self._source = FileStack(filetype, path, *args, **kwargs)
        else:
            self._source = None
        self._source = self.comm.bcast(self._source)

        # compute the size; start with full file.
        lstart = self.comm.rank * self._source.size // self.comm.size
        lend = (self.comm.rank  + 1) * self._source.size // self.comm.size
        self._size = lend - lstart

        self.start = 0
        self.end = self._source.size

        self._lstart = lstart # offset in the file for this rank
        self._lend = lend     # offset in the file for this rank

        # update the meta-data
        self.attrs.update(self._source.attrs)

        if self.comm.rank == 0:
            self.logger.info("Extra arguments to FileType: %s %s" % (str(args), str(kwargs)))

        CatalogSource.__init__(self, comm=comm)

    def query_range(self, start, end):
        """
            Seek to a range in the file catalog.

            Parameters
            ----------
            start : int
                start of the file relative to the physical file

            end : int
                end of the file relative to the physical file

            Returns
            -------
            A new catalog that only accesses the given region of the file.

            If the original catalog (self) contains any assigned columns not directly
            obtained from the file, then the function will raise ValueError, since
            the operation in that case is not well defined.

        """
        if len(CatalogSource.hardcolumns.fget(self)) > 0:
            raise ValueError("cannot seek if columns have been attached to the FileCatalog")

        other = self.copy()
        other._lstart = self.start + start +  self.comm.rank * (end - start) // self.comm.size
        other._lend = self.start + start + (self.comm.rank + 1) * (end - start) // self.comm.size
        other._size = other._lend - other._lstart
        other.start = start
        other.end = end
        CatalogSource.__init__(other, comm=self.comm)
        return other

    def __repr__(self):
        path = self._source.path
        name = self.__class__.__name__
        args = (name, self.size, repr(self._source))

        return "%s(size=%d, %s)" % args

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
            return self._source.get_dask(col)[self._lstart:self._lend]
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
    def __init__(self, path, *args, **kwargs):
        comm = kwargs.pop('comm', None)
        attrs = kwargs.pop('attrs', {})
        FileCatalogBase.__init__(self, filetype=filetype, path=path, args=args, kwargs=kwargs, comm=comm)
        self.attrs.update(attrs)

    # make the doc string for this class
    __doc__ = _make_docstring(filetype, examples)

    # make the new class object and return it
    newclass = type(name, (FileCatalogBase,),{"__init__": __init__, "__doc__":__doc__})
    return newclass


class FileCatalog(FileCatalogBase):
    """
    Base class to create a source of particles from a
    single file, or multiple files, on disk.

    Files of a specific type should be subclasses of this class.

    Parameters
    ----------
    filetype : subclass of :class:`~nbodykit.io.base.FileType`
        the file-like class used to load the data from file; should be a
        subclass of :class:`nbodykit.io.base.FileType`
    path : string or list. If string, it is expanded as a glob pattern.
    attrs : dict, attributes to set to the Catalog.
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
    def __init__(self, filetype, path, *args, **kwargs): 
        comm = kwargs.pop('comm', None)
        attrs = kwargs.pop('attrs', {})
        FileCatalogBase.__init__(self, filetype=filetype, path=path, args=args, kwargs=kwargs, comm=comm)
        self.attrs.update(attrs)

CSVCatalog       = FileCatalogFactory("CSVCatalog", io.CSVFile, examples='csv-data')
BinaryCatalog    = FileCatalogFactory("BinaryCatalog", io.BinaryFile, examples='binary-data')
BigFileCatalog   = FileCatalogFactory("BigFileCatalog", io.BigFile, examples='bigfile-data')
HDFCatalog       = FileCatalogFactory("HDFCatalog", io.HDFFile, examples='hdf-data')
TPMBinaryCatalog = FileCatalogFactory("TPMBinaryCatalog", io.TPMBinaryFile)
FITSCatalog      = FileCatalogFactory("FITSCatalog", io.FITSFile, examples='fits-data')
Gadget1Catalog   = FileCatalogFactory("Gadget1Catalog", io.Gadget1File, examples=None)
