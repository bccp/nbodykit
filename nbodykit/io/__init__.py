from abc import abstractmethod, abstractproperty
import numpy
from ..extern.six import string_types
from ..plugins import PluginBase

class FileType(PluginBase):
    """
    Abstract base class representing a file object
    """ 
    required_attributes = ['size', 'dtype'] 
        
    @abstractmethod
    def read_chunk(self, columns, start, stop, step=1):
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
    def ncol(self):
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
        
    def _slice_size(self, start, stop, step):
        N, remainder = divmod(stop-start, step)
        if remainder: N += 1
        return N
    
    def __getitem__(self, s):
        """
        Return a slice of the data, indexed in 
        array-like fashion
        """
        if isinstance(s, string_types): s = [s]
        
        # slicing with valid columns, returns a "view" of the current file
        # with the different dtype
        if isinstance(s, list):
            if not all(isinstance(k, string_types) for k in s):
                raise ValueError("string keys should be one of %s" %str(self.keys()))
            if not all(ss in self.keys() for ss in s):
                invalid = [col for col in s if s not in self.keys()]
                raise ValueError("invalid string keys: %s; run keys() for valid options" %str(invalid))
            
            obj = object.__new__(self.__class__)
            obj.dtype = numpy.dtype([(col, self.dtype[col]) for col in s])
            obj.size = self.size
            obj.base = self # set the base that owns the memory
            return obj
        
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
        
        # if we don't own memory, return from base
        if getattr(self, 'base', None) is None:
            return self.read(self.keys(), start, stop, step)
        else:
            return self.base.read(self.keys(), start, stop, step)
        
    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range,
        as a dictionary

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the file (in particles)
        """
        # columns should be a list
        if isinstance(columns, string_types): columns = [columns]

        # initialize the return array
        N, remainder = divmod(stop-start, step)
        if remainder: N += 1
        dtype = [(col, self.dtype[col]) for col in columns]
        toret = numpy.empty(N, dtype=dtype)

        # return each column requested
        i = 0
        for chunk in self.read_chunk(columns, start, stop, step=step):
            N = len(chunk)
            for column in columns:
                toret[column][i:i+N] = chunk[column][:]
            i += N 
            
        return toret

def io_extension_points():
    
    return {'FileType' : FileType}