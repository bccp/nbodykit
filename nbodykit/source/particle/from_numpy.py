from nbodykit.base.particles import ParticleSource
from nbodykit import CurrentMPIComm
import numpy

class Array(ParticleSource):
    """
    A source of particles from numpy array
    """
    @CurrentMPIComm.enable
    def __init__(self, data, comm=None, use_cache=False, **kwargs):
        """
        Parameters
        ----------
        comm : MPI.Communicator
            the MPI communicator
        data : numpy.array
            a structured array holding the 
        
        """
        self.comm    = comm
        self._source = data
        if data.dtype.names is None:
            raise ValueError("input data should be a structured numpy array")
        
        # verify data types are the same
        dtypes = self.comm.gather(data.dtype, root=0)
        if self.comm.rank == 0:
            if any(dt != dtypes[0] for dt in dtypes):
                raise ValueError("mismatch between dtypes across ranks in Array")
        
        # update the meta-data
        self.attrs.update(kwargs)

        ParticleSource.__init__(self, comm=comm, use_cache=use_cache)

    @property
    def size(self):
        return len(self._source)
        
    @property
    def hardcolumns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        defaults = ParticleSource.hardcolumns.fget(self)
        return list(self._source.dtype.names) + defaults

    def get_hardcolumn(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        if col in self._source.dtype.names: 
            return self.make_column(self._source[col])
        else:
            return ParticleSource.get_hardcolumn(self, col)

