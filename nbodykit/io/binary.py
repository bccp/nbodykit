import numpy
import os

from . import FileType, tools
from ..extern.six import string_types

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
    a binary file. 
        
    .. warning::
        
        This assumes the data is stored in a column-major format
    """
    plugin_name = "BinaryFile"
    
    def __init__(self, path, dtype, offsets=None, header_size=0, size=None):
                
        # the file path
        self.path = path
        
        # set the data type
        self.dtype = dtype
        if not isinstance(self.dtype, numpy.dtype):
            self.dtype = numpy.dtype(self.dtype)
                                
        # determine the size (either an int or a function)
        if size is None:
            size = lambda fn: getsize(fn, header_size, self.dtype.itemsize)
        if callable(size):
            self.size = size(self.path)
        elif ininstance(size, int):
            self.size = size
        else:
            raise TypeError("`size` keyword should be a callable or integer")
        
        # use the input offsets dict
        if offsets is not None:
            if not isinstance(offsets, dict):
                raise TypeError("`offsets` keyword should be a dict")
            self.offsets = offsets.copy()
            
            # make sure each column in dtype is in the offsets table
            if not all(col in self.offsets for col in self):
                raise ValueError("missing some dtype columns in the input `offsets` dict")
        # create the dictionary of offsets
        else:            
            self.offsets = {}
            for col in self:
                self.offsets[col] = self._default_byte_offset(col, header_size=header_size)
        
        # for returning views of the file
        self.base = None
        
    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "a binary file reader"
        
        s.add_argument("path", type=str, 
            help='the name of the binary file to load')
        s.add_argument("dtype", nargs='+', type=tuple, 
            help='list of tuples of (name, dtype) to be converted to a numpy.dtype')
        s.add_argument("header_size", type=int,
            help='the size of the header of the in bytes')
        s.add_argument("size",
            help=("an int giving the file size or a function that takes a single argument, "
                  "the name of the file, and returns the size"))
        s.add_argument("offsets", type=dict,
            help='a dictionary giving the byte offsets for each column in the file')
        
    def _default_byte_offset(self, col, header_size=0):
        """
        Internal function to return the offset in bytes
        for the column name
        
        This assumes consecutive storage of columns, so the offset
        for the second column is the size of the full array of the 
        first column (plus header size)
        """
        offset = header_size
        cols = self.keys()
        i = 0
        while i < cols.index(col):
            offset += self.size*self.dtype[cols[i]].itemsize
            i += 1
        
        return offset
        
    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range, 
        as a dictionary
        
        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the binary file (in particles)
        """ 
        if isinstance(columns, string_types): columns = [columns]
        
        dt = [(col, self.dtype[col]) for col in columns]
        toret = numpy.empty(tools.get_slice_size(start, stop, step), dtype=dt)
               
        with open(self.path, 'rb') as ff:
            
            for col in columns:
                offset = self.offsets[col]
                dtype = self.dtype[col]
                ff.seek(offset, 0)
                ff.seek(start * dtype.itemsize, 1)
                toret[col][:] = numpy.fromfile(ff, count=stop-start, dtype=dtype)[::step]
    
        return toret