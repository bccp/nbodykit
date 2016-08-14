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
        
    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range,
        as a dictionary

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the file (in particles)
        """
        if isinstance(columns, string_types): columns = [columns]

        # initialize the return array
        N, remainder = divmod(stop-start, step)
        if remainder: N += 1
        toret = numpy.empty(N, dtype=self.dtype)

        # loop over slices
        global_start = 0
        for fnum in self._file_range(start, stop):

            # the slice
            sl = self._normalized_slice(start, stop, fnum)
            
            # do the reading
            tmp = self.files[fnum].read(column, sl[0], sl[1], step)
            for col in columns:
                toret[col][global_start:global_start+len(tmp)] = tmp[col][:]

            global_start += len(tmp) # update where we start slicing return array
            
        return toret[columns]
    
            
        
        
        
