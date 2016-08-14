from abc import abstractmethod, abstractproperty
from ..plugins import PluginBase

class FileType(PluginBase):
    """
    Abstract base class representing a file object
    """         
    @abstractproperty
    def size(self):
        """
        The total number of rows in the file
        """
        pass

    @abstractproperty
    def dtype(self):
        """
        The ``numpy.dtype`` of the data stored in
        the file
        """
        pass
        
    @abstractmethod
    def read(self, columns, start, stop, step=1):
        """
        Read the specified columns from the file from 
        `start` to `stop` with a stepsize of `step`
        
        Parameters
        ----------
        columns : str, list of str
            the columns to be read
        start : int
            the row integer to start reading at
        stop : int
            the row integer to stop reading at
        step : int, optional
            the step size to use when reading; default is 1
        
        Returns
        -------
        data : array_like
            a numpy structured array holding the requested data
        """
        pass
    
    @property
    def ncols(self):
        """
        The number of data columns in the file
        """
        return len(self.dtype)
    
    @property
    def shape(self):
        """
        The shape of the file, equal to the size of
        the file in axis 0
        """
        return (self.size,)
    
    def __len__(self):
        return self.size
      
    def __iter__(self):
        return iter(self.keys())
    
    def keys(self):
        return [k for k in self.dtype.names]
        
    def __getitem__(self, s):
        """
        Return a slice of the data, indexed in 
        array-like fashion
        """
        # input is tuple
        if isinstance(s, tuple): s = s[0]
        
        # input is integer
        if isinstance(s, int):
            start, stop, step = s, s+1, 1
        # input is a slice
        elif isinstance(s, slice):
            start, stop, step = s.indices(self.size)
        else:
            raise IndexError("FileType index should be an integer or slice")
            
        return self.read(self.keys(), start, stop, step)

def io_extension_points():
    
    return {'FileType' : FileType}