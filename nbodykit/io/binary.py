import os
import numpy
from . import FileType

def getsize(filename, header_size, rowsize):
    """
    The default method to determine the size of the binary file
    
    The "size" is defined as the number of rows, where each
    row has of size of `rowsize` in bytes.
    
    Notes
    -----
    *   This assumes the input file is not compressed
    *   This function does not depend on the layout of the
        binary file, i.e., if the data is formatted in actual
        rows or not
    
    Raises
    ------
    ValueError : 
        If the function determines a fractional number of rows
    
    Parameters
    ----------
    filename : str
        the name of the binary file
    header_size : int
        the size of the header in bytes, which will be skipped
        when determining the number of rows
    rowsize : int
        the size of the data in each row in bytes
    """
    bytesize = os.path.getsize(filename)
    size, remainder = divmod(bytesize-header_size, rowsize)
    if remainder != 0:
        raise ValueError("byte size mismatch -- fractional rows found")
    return size
    

class BinaryFile(FileType):
    """
    A file object to handle the reading of columns of data from 
    a binary file
    
    .. warning::
        
        This assumes the data is stored in a column-major format
    """
    plugin_name = "BinaryFile"
    
    def __init__(self, path, dtype, header_size=0, peek_size=None):
                
        # function to determine the size
        if self.peek_size is None:
            self.peek_size = lambda fn: getsize(fn, self.header_size, self.dtype.itemsize)
        
    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "a binary file reader"
        
        s.add_argument("path", type=str, 
            help='the name of the binary file to load')
        s.add_argument("dtype", nargs='+', type=tuple, 
            help='list of tuples of (name, dtype) to be converted to a numpy.dtype')
        s.add_argument("header_size", type=int,
            help='the size of the header of the in bytes')
        s.add_argument("peek_size", 
            help=("a function taking a single argument, the name of the file, "
                  "and returns the true size of each file"))
                    
    @property
    def dtype(self):
        """The data type"""
        return self._dtype
        
    @dtype.setter
    def dtype(self, val):
        self._dtype = val
        if not isinstance(self._dtype, numpy.dtype):
            self._dtype = numpy.dtype(self._dtype)
        
    @property
    def size(self):
        """The size of the file"""
        try:
            return self._size
        except AttributeError:
            self._size = self.peek_size(self.path)
            return self._size
        
    def _offset(self, col):
        """
        Internal function to return the offset in bytes
        for the column name
        
        This assumes consecutive storage of columns, so the offset
        for the second column is the size of the full array of the 
        first column (plus header size)
        """
        offset = self.header_size
        cols = self.keys()
        i = 0
        while i < cols.index(col):
            offset += self.size*self.dtype[cols[i]].itemsize
            i += 1
        
        return offset
        
    def read_chunk(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range, 
        as a dictionary
        
        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the binary file (in particles)
        """ 
        dt = [(col, self.dtype[col]) for col in columns]
        toret = numpy.empty(self._slice_size(start, stop, step), dtype=dt)
               
        with open(self.path, 'rb') as ff:
            
            for col in columns:
                offset = self._offset(col)
                dtype = self.dtype[col]
                ff.seek(offset, 0)
                ff.seek(start * dtype.itemsize, 1)
                toret[col][:] = numpy.fromfile(ff, count=stop - start, dtype=dtype)[::step]
    
        yield toret
            
        
        
        
