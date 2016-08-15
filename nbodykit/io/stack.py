from glob import glob
import os
import numpy
from ..extern.six import string_types
from . import FileType

class FileStack(FileType):
    """
    A class that offers a continuous view of a stack of 
    subclasses of :class:`FileType` instances
    """
    plugin_name = "FileStack"
    
    def __init__(self, path, filetype, **kwargs):
        
        # save the list of relevant files
        if isinstance(path, list):
            filenames = path
        elif isinstance(path, string_types):
        
            if '*' in path:
                filenames = list(map(os.path.abspath, sorted(glob(path))))
            else:
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                filenames = [os.path.abspath(path)]
        else:
            raise ValueError("'path' should be a string or a list of strings")
            
        self.files = [filetype(fn, **kwargs) for fn in filenames]
        self.sizes = numpy.array([len(f) for f in self.files], dtype='i8')
                
    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "a consecutive view of a stack of files"
        
        s.add_argument("path", type=str, nargs='*',
            help='a list of files or a glob-like pattern')
        s.add_argument("filetype", 
            help='the file type class')
                    
    @classmethod
    def from_files(cls, files, sizes=[]):
        
        stack = object.__new__(cls)
        stack.files = files
        if len(sizes):
            if len(sizes) != len(stack.files):
                raise ValueError("length mismatch: `sizes` should specify the size")
            stack.sizes = numpy.asarray(sizes, dtype='i8')
        else:
            stack.sizes = numpy.array([len(f) for f in stack.files], dtype='i8')
        
        return stack
            
        
    @property
    def dtype(self):
        return self.files[0].dtype
        
    @property
    def size(self):
        return self.cumsizes[-1]
        
    @property
    def cumsizes(self):
        """
        Cumulative size counts across all files
        """
        try:
            return self._cumsizes
        except AttributeError:
            self._cumsizes = numpy.zeros(self.nfiles+1, dtype=self.sizes.dtype)
            self._cumsizes[1:] = self.sizes.cumsum()
            return self._cumsizes
        
    @property
    def nfiles(self):
        return len(self.files)

    def _file_range(self, start, stop):
        """
        Convert global to local indices
        """
        fnums = numpy.searchsorted(self.cumsizes[1:], [start, stop])
        return list(range(fnums[0], fnums[1]+1))
        
    def _normalized_slice(self, start, stop, fnum):
        """
        Convert global to local indices
        """
        start_size = self.cumsizes[fnum] 
        return (max(start-start_size, 0), min(stop-start_size, self.sizes[fnum]))
        
    def read_chunk(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range,
        as a dictionary

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the file (in particles)
        """
        # loop over the files we need to read from
        for fnum in self._file_range(start, stop):

            # the local slice
            sl = self._normalized_slice(start, stop, fnum)
            
            # yield this chunk
            return self.files[fnum].read_chunk(columns, sl[0], sl[1], step)


    
            
        
        
        