from nbodykit.io.stack import FileStack
from nbodykit.base.particles import ParticleSource
from nbodykit import CurrentMPIComm
import numpy

class File(ParticleSource):
    """
    Read a source of particles from a single file, or multiple
    files, on disk
    """        
    @CurrentMPIComm.enable
    def __init__(self, filetype, path, args={}, comm=None, use_cache=False, **kwargs):
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
        **kwargs : 
            additional keywords are stored as meta-data in the :attr:`attrs` dict
        """
        self.comm = comm
        self.filetype = filetype
        
        # bcast the FileStack
        if self.comm.rank == 0:
            self._source = FileStack(path, filetype, **args)
        else:
            self._source = None
        self._source = self.comm.bcast(self._source)

        # update the meta-data
        self.attrs.update(self._source.attrs)
        self.attrs.update(kwargs)

        if self.comm.rank == 0:
            self.logger.info("Extra arguments to FileType: %s" %args)

        ParticleSource.__init__(self, comm=comm, use_cache=use_cache)

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
        return list(self._source.dtype.names)

    def get_hardcolumn(self, col):
        """
        Return a column from the underlying file source

        Columns are returned as dask arrays
        """
        start = self.comm.rank * self._source.size // self.comm.size
        end = (self.comm.rank  + 1) * self._source.size // self.comm.size
        return self._source.get_dask(col)[start:end]

