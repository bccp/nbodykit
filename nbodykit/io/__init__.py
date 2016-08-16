from ..extern.six import string_types
from ..plugins import PluginBase

import numpy
from abc import abstractmethod, abstractproperty

def get_slice_size(start, stop, step):
    """
    Return the size of an array slice
    
    Parameters
    ----------
    start : int
        the beginning of the slice
    stop : int
        the end of the slice
    step : int
        the slice step size
    
    Returns
    -------
    N : int
        the total size of the slice
    """
    N, remainder = divmod(stop-start, step)
    if remainder: N += 1
    return N

class FileType(PluginBase):
    """
    Abstract base class representing a file object
    """ 
    required_attributes = ['size', 'dtype'] 
    
    @abstractmethod
    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range,
        returning a structured numpy array

        Parameters
        ----------
        columns : str, list of str
            the name of the column(s) to return
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
        """
        Return the list of the fields in :attr:`dtype`
        """
        return [k for k in self.dtype.names]
            
    def __getitem__(self, s):
        """
        This function provides numpy-like array indexing
        
        It supports:
        
            1.  integer, slice-indexing similar to arrays
            2.  string indexing providing names in :func:`keys`
        
        .. note::
        
            If a single column is being returned, a numpy array 
            holding the data is returned, rather than a structured
            array with only a single field.
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
            if s < 0: s += self.size
            start, stop, step = s, s+1, 1
        # input is a slice
        elif isinstance(s, slice):
            start, stop, step = s.indices(self.size)
        else:
            raise IndexError("FileType index should be an integer or slice")
        
        # if we don't own memory, return from base
        if getattr(self, 'base', None) is None:
            toret = self.read(self.keys(), start, stop, step)
        else:
            toret = self.base.read(self.keys(), start, stop, step)
            
        # if structured array only has single field, return
        # the numpy array for that field
        if len(toret.dtype) == 1:
            toret = toret[toret.dtype.names[0]]
            
        return toret

def io_extension_points():
    """
    Return a dictionary of the extension points for :mod:`io`
    
    This returns only the :class:`FileType` class
    """
    return {'FileType' : FileType}