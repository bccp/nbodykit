from glob import glob
import os
import numpy
from ..extern.six import string_types
from .filetype import FileTypeBase

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
    

class BinaryFile(FileTypeBase):
    """
    A file object to handle the reading of columns of data from 
    a binary file
    
    .. warning::
        
        This assumes the data is stored in a column-major format
    """
    def __init__(self, path, dtype, header_size=0, peek_size=None):
        """
        Parameters
        ----------
        path : str, list of str
            either a list of filenames, a single filename, or a 
            glob-like pattern to match files on
        dtype : list of tuples, numpy.dtype
            the data types of the data stored in the binary file
        header_size : int, optional
            the size of the header of each file in bytes
        peek_size : callable, optional
            a function taking a single argument, the name of the file, 
            and returns the number of true "size" of each file
        """
        # save the list of relevant files
        if isinstance(path, list):
            self.filenames = path
        elif isinstance(path, string_types):
        
            if '*' in path:
                self.filenames = list(map(os.path.abspath, sorted(glob(path))))
            else:
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                self.filenames = [os.path.abspath(path)]
        else:
            raise ValueError("'path' should be a string or a list of strings")
            
        # construct the dtype
        self.dtype = dtype
        if not isinstance(self.dtype, numpy.dtype):
            self.dtype = numpy.dtype(dtype)
    
        # determine the sizes
        if peek_size is None:
            peek_size = lambda fn: getsize(fn, header_size, self.dtype.itemsize)
        self.sizes = numpy.array([peek_size(fn) for fn in self.filenames])
        
        # header size    
        self.header_size = header_size
    
    def __getitem__(self, s):
        if isinstance(s, tuple): s = s[0]
        start, stop, step = s.indices(self.size)
        return self.read(self.keys(), start, stop, step)   
    
    def __iter__(self):
        return iter(self.keys())
    
    def keys(self):
        return list(self.dtype.names)

    @property
    def shape(self):
        return (self.size,)
        
    @property
    def ncols(self):
        return len(self.dtype)
    
    @property
    def nfiles(self):
        return len(self.filenames)
        
    @property
    def size(self):
        return self.sizes.sum()
        
    def _offset(self, fnum, col):
        """
        Internal function to return the offset (per file) in bytes
        for the specified file number and column name
        
        This assumes consecutive storage of columns, so the offset
        for the second column is the size of the full array of the 
        first column (plus header size)
        """
        offset = self.header_size
        cols = self.keys()
        i = 0
        while i < cols.index(col):
            offset += self.sizes[fnum]*self.dtype[cols[i]].itemsize
            i += 1
        
        return offset
        
    def _read_block(self, col, fnum, start, stop, step=1):
        """
        Internal read function that reads the specified block
        of bytes and formats appropriately
        
        Parameters
        ----------
        col : str
            the name of the column we are reading
        fnum : int
            the file number to open
        start : int
            the start position in particles, ranging from 0 to the
            size of the open file
        stop : int 
            the stop position in particiles, ranging from 0 to 
            the size of the open file
        """
        with open(self.filenames[fnum], 'rb') as ff:
            offset = self._offset(fnum, col)
            dtype = self.dtype[col]
            ff.seek(offset, 0)
            ff.seek(start * dtype.itemsize, 1)
            return numpy.fromfile(ff, count=stop - start, dtype=dtype)[::step]
        
        
    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range, 
        as a dictionary
        
        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the binary file (in particles)
        """
        if isinstance(columns, string_types): columns = [columns]
            
        # make the slices positive
        if start < 0: start += self.size
        if stop < 0: stop += self.size
        
        # determine the file numbers of start/stop from cumulative sizes
        cumsum = numpy.zeros(self.nfiles+1, dtype=self.sizes.dtype)
        cumsum[1:] = self.sizes.cumsum()
        fnums = numpy.searchsorted(cumsum[1:], [start, stop])
          
        # initialize the return array
        N, remainder = divmod(stop-start, step)
        if remainder: N += 1
        toret = numpy.empty(N, dtype=self.dtype)
          
        # loop over necessary files to get requested range
        global_start = 0
        for fnum in range(fnums[0], fnums[1]+1):
                        
            # normalize the global start/stop to the per file values     
            start_size = cumsum[fnum] 
            this_slice = (max(start-start_size, 0), min(stop-start_size, self.sizes[fnum]), step)
            
            # do the reading
            for col in columns:
                tmp = self._read_block(col, fnum, *this_slice)
                toret[col][global_start:None] = tmp[:]
                
            global_start += len(tmp) # update where we start slicing return array

        return toret[columns]
                                    
if __name__ == '__main__':
    
    import os
    import dask.array as da
    
    # file path
    if 'cori' in os.environ['HOSTNAME']:
        path = "/global/cscratch1/sd/nhand/Data/RunPBDM/PB00/tpmsph_0.5000.bin.*"
    else:
        path = "/scratch2/scratchdirs/nhand/Data/RunPBDM/PB00/tpmsph_0.5000.bin.*"
    
    # dtype of files
    dtypes = [('Position', ('f4', 3)), ('Velocity', ('f4', 3)), ('ID', 'u8')]
    
    # open the binary file
    f = BinaryFile(path, dtypes, header_size=28)
    
    # get the dask array straight from file (this is awesome)
    data = da.from_array(f, chunks=1024*1024)
    
    # mean out mean of first 1000 particles
    meanop = data[:1000]['Position'].mean(axis=0)
    print(meanop.compute())
    
            
        
        
        
