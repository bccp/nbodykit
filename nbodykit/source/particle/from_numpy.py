from nbodykit.base.particles import ParticleSource
from nbodykit import CurrentMPIComm
import numpy

class Array(ParticleSource):
    """
    A source of particles from numpy array
    """
    @CurrentMPIComm.enable
    def __init__(self, data, BoxSize, Nmesh, comm=None, **kwargs):
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
                raise ValueError("mismatch between dtypes across ranks in ParticlesFromNumpy")
        self.dtype = data.dtype

        # update the meta-data
        self.attrs.update(kwargs)

        ParticleSource.__init__(self, BoxSize=BoxSize, Nmesh=Nmesh, dtype='f4', comm=comm)

    def get_column(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        import dask.array as da
        return da.from_array(self._source[col], chunks=100000)

    @property
    def size(self):
        return len(self._source)
        
    @property
    def hcolumns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        return list(self._source.dtype.names)
