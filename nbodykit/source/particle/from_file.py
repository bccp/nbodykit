from nbodykit.io.stack import FileStack
from nbodykit.base.particles import ParticleSource
from nbodykit.base.painter import Painter
from nbodykit import CurrentMPIComm
import numpy

class ParticlesFromFile(ParticleSource):
    """
    Read particles from file
    """
    @CurrentMPIComm.enable
    def __init__(self, filetype, path, args={}, comm=None, **kwargs):
        """
        Parameters
        ----------
        comm : MPI.Communicator
            the MPI communicator
        filetype : class
            the type of class to load 
        path : str
            the path to the file
        """
        self.comm = comm
        self._source = FileStack(path, filetype, **args)

        # update the meta-data
        self.attrs.update(self._source.attrs)
        self.attrs.update(kwargs)

        if self.comm.rank == 0:
            self.logger.info("Extra arguments to FileType: %s " % args)

        ParticleSource.__init__(self, comm)

    def __getitem__(self, col):
        """
        Return a column from the underlying file source

        Columns are returned as dask arrays
        """
        if col in self._source:
            start = self.comm.rank * self._source.size // self.comm.size
            end = (self.comm.rank  + 1) * self._source.size // self.comm.size
            return self._source.get_dask(col)[start:end]

        return ParticleSource.__getitem__(self, col)

    @property
    def size(self):
        """
        The local size
        """
        start = self.comm.rank * self._source.size // self.comm.size
        end = (self.comm.rank  + 1) * self._source.size // self.comm.size
        return end - start

    @property
    def hcolumns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        return list(self._source.dtype.names)

