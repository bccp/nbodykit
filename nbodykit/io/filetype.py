import abc
from ..extern.six import add_metaclass

@add_metaclass(abc.ABCMeta)
class FileTypeBase(object):
    """
    File type extension point
    """ 
    @abc.abstractmethod
    def __getitem__(self, s):
        pass
        
    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractproperty
    def dtype(self):
        pass
    
    @property
    def ncols(self):
        return len(self.dtype)
    
    @property
    def shape(self):
        return (self.size,)
    
    def __len__(self):
        return self.size
      
    def __iter__(self):
        return iter(self.keys())
    
    def keys(self):
        return [k for k in self.dtype.names]
    