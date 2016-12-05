from nbodykit.io.stack import FileStack
from nbodykit.base.particles import ParticleSource
from nbodykit.base.painter import Painter
from nbodykit import CurrentMPIComm
import numpy

class ParticlesFromNumpy(ParticleSource):
    """
    A source of particles from numpy array
    """
    @CurrentMPIComm.enable
    def __init__(self, data, comm=None, **kwargs):
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
        
        # the total collective size
        self._csize = self.comm.allreduce(len(data))
        
        # verify data types are the same
        dtypes = self.comm.gather(data.dtype, root=0)
        if self.comm.rank == 0:
            if any(dt != dtypes[0] for dt in dtypes):
                raise ValueError("mismatch between dtypes across ranks in ParticlesFromNumpy")
        self.dtype = data.dtype
        
        # update the meta-data
        self.attrs.update(kwargs)
        if self.comm.rank == 0:
            self.logger.info("attrs = %s" % self.attrs)

    def __getitem__(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        if col in self.transform:
            return self.transform[col](self)
        elif col in self._source.dtype.names:
            import dask.array as da
            return da.from_array(self._source[col], chunks=100000)
        else:
            raise KeyError("column `%s` is not a valid column name" %col)
        
    @property
    def size(self):
        return len(self._source)
        
    @property
    def csize(self):
        return self._csize
        
    @property
    def columns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        return sorted(set(list(self._source.dtype.names) + list(self.transform)))

    def read(self, columns):
        """
        Return the requested columns as dask arrays
        
        Currently, this returns a dask array holding the total amount
        of data for each rank, divided equally amongst the available ranks
        """
        return [self[col] for col in columns]

    def paint(self, pm):
        """
        Paint to the mesh
        """
        # paint and apply any transformations to the real field
        real = self.painter(self, pm)
        self.painter.transform(self, real)
        
        return real
