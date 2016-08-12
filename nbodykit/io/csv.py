import numpy
import pandas as pd
import os

from .filetype import FileTypeBase
from ..extern.six import string_types
from .utils import consecutive_view_slices
from .filetype import FileTypeBase

def get_partition_sizes(filename, blocksize):
    """
    From a filename and preferred blocksize in bytes, 
    return the number of rows in each partition 
    
    This divides the input file into partitions with size
    roughly equal to blocksize, reads the bytes, and counts
    the number of delimiter
    """  
    from dask.bytes.local import read_block_from_file

    # search for lines separated by newline char
    delimiter = "\n".encode()
    
    # size in bytes and byte offsets of each partition
    size = os.path.getsize(filename)
    offsets = list(range(0, size, int(blocksize)))
    
    nrows = []
    for offset in offsets:
        block = read_block_from_file(filename, offset, blocksize, delimiter, False)
        nrows.append(block.count(delimiter))
    return nrows

def infer_dtype(path, names, config):
    """
    Read the first few lines of the CSV to determine the 
    data type
    """
    # read the first line to get the the dtype
    df = pd.read_csv(path, nrows=1, names=names, **config)
    
    toret = {}
    for name in names:
        toret[name] = df[name].dtype
    return toret


class CSVFile(FileTypeBase):
    """
    A file object to handle the reading of columns of data from 
    a CSV file
    
    Internally, this class uses :func:`pandas.read_csv` and supports 
    all of the keyword arguments.
    
    .. warning:: 
    
        This assumes the delimiter for separate lines is the newline
        character.
    """
    plugin_name = 'CSVFile'
    
    def __init__(self, path, names, blocksize=32*1024*1024, dtype={}, delim_whitespace=True, header=None, **config):
        """
        Parameters
        ----------
        path : str
            the name of the file we are reading from
        names : list of str
            list of the names of each column in the file to read
        dtype : {str, dict}, optional
            the data type of individual columns; if not provided, the 
            data types are inferred directly from the data
        delim_whitespace : bool, optional
            set to `True` if the CSV file is space-separated
        **kwargs: dict
            options to pass down to :func:`pandas.read_csv`
        """
        self.path = path
        self._names = names
        self.blocksize = blocksize
        
        # set the read_csv defaults
        if 'sep' in config or 'delimiter' in config:
            delim_whitespace = False
        config.setdefault('delim_whitespace', delim_whitespace)
        config.setdefault('header', header)
        config.setdefault('engine', 'c')
        self._config = config
        
        # dtype can also be a string --> apply to all columns
        if isinstance(dtype, string_types):
            dtype = {col:dtype for col in names}
        
        # infer the data type?
        if not all(col in dtype for col in names):
            inferred_dtype = infer_dtype(path, names, config)
        
        # store the dtype as a list
        self.dtype = []
        for col in names:
            if col in dtype:
                dt = dtype[col]
                if not isinstance(dt, numpy.dtype):
                    dt = numpy.dtype(dt)
            else:
                dt = inferred_dtype[col]
            self.dtype.append((col, dt))
    
    @property
    def dtype(self):
        return self._dtype
    
    @dtype.setter
    def dtype(self, val):
        self._dtype = val
        
    def __getitem__(self, s):
        if isinstance(s, tuple): s = s[0]
        start, stop, step = s.indices(self.size)
        return self.read(self.keys(), start, stop, step)   
    
    def __iter__(self):
        return iter(self.keys())
    
    def keys(self):
        return list(self._names)

    def __len__(self):
        return self.size
        
    @property
    def shape(self):
        return (self.size,)
        
    @property
    def ncols(self):
        """
        The total number of columns in the file
        """
        return len(self.dtype)
    
    @property
    def sizes(self):
        """
        The sizes of the individual partitions
        """        
        try: 
            return self._sizes
        except AttributeError:
            self._sizes = get_partition_sizes(self.path, self.blocksize)
            return self._sizes
   
    @property
    def size(self):
        """
        The total size of the file (equivalent to the total 
        number of rows)
        """
        return sum(self.sizes)

    @property
    def _data(self):
        """
        The dask dataframe
        """
        try:
            return self._dask_data
        except: 
            import dask.dataframe as dd       
            
            kws = self._config.copy()
            kws['dtype'] = dict(self.dtype)
            kws['names'] = self.keys()
            kws['blocksize'] = self.blocksize
            self._dask_data = dd.read_csv(self.path, **kws)
            if self._dask_data.npartitions != len(self.sizes):
                raise ValueError("bad bad bad")
            return self._dask_data
        
    def _read_block(self, cols, fnum, start, stop, step=1):
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
        get = getattr(self._data, 'get_partition', getattr(self._data, 'get_division'))
        data = get(fnum).compute()
        return data.to_records(index=False)[start:stop:step]
            
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
        for partnum, sl in consecutive_view_slices(start, stop, step, self.sizes):
            
            # do the reading
            tmp = self._read_block(columns, partnum, *sl)
            for col in columns:
                toret[col][global_start:global_start+len(tmp)] = tmp[col][:]
            
            global_start += len(tmp) # update where we start slicing return array

        return toret[columns]
                                    
if __name__ == '__main__':

    import dask.array as da
    
    # file path
    path = "/Users/nhand/Library/Caches/nbodykit/data/test_bianchi_data.dat"

    # names of the columns
    names = ['ra', 'dec', 'z']
    f = CSVFile(path, names, delim_whitespace=True, dtype='f4')
    
    # get the dask array straight from file (this is awesome)
    data = da.from_array(f, chunks=100)
    
    # mean out mean of first 1000 particles
    meanop = data[:500]['ra'].mean()
    print(meanop.compute())
    
            
        
        
        
