from nbodykit.io.stack import FileStack
from nbodykit.base.particles import ParticleSource
from nbodykit.base.painter import Painter

import numpy

class ParticlesFromFile(ParticleSource):
    """
    Read particles from file
    """
    def __init__(self, comm, filetype, path, args={}, **kwargs):
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
            self.logger.info("attrs = %s" % self.attrs)
            
    def __getitem__(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        if col in self.transform:
            return self.transform[col](self)
        elif col in self._source:
            return self._source.get_dask(col)
        else:
            raise KeyError("column `%s` is not a valid column name" %col)
        
    @property
    def size(self):
        """
        The local size
        """
        size = self.csize // self.comm.size
        if self.comm.rank == self.comm.size-1:
            size += self.csize % self.comm.size
        return size
        
    @property
    def csize(self):
        """
        The collective size
        """
        return self._source.size
        
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
        start = self.comm.rank * self.csize // self.comm.size
        end = (self.comm.rank  + 1) * self.csize // self.comm.size
        
        return [self[col][start:end] for col in columns]

    def paint(self, pm):
        """
        Paint to the mesh
        """
        # paint and apply any transformations to the real field
        real = self.painter(self, pm)
        self.painter.transform(self, real)
        return real