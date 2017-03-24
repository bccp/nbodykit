import numpy
import os
from pandas import read_csv 
    
from ..extern.six import string_types
from .base import FileType
from . import tools

class CSVPartition(object):
    """
    A simple class to convert byte strings of data from a CSV file
    to a pandas DataFrame on demand
    
    The DataFrame is cached as :attr:`value`, so only a single
    call to :func:`pandas.read_csv` is used
    """
    def __init__(self, filename, offset, blocksize, delimiter, **config):
        """
        Parameters
        ----------
        filename : str
            the file to read data from
        offset : int 
            the offset in bytes to start reading at
        blocksize : int
            the size of the bytes block to read
        delimiter : byte str
            how to distinguish separate lines
        config : 
            the configuration keywords passed to :func:`pandas.read_csv`
        """
        self.filename  = filename
        self.offset    = offset
        self.blocksize = blocksize
        self.delimiter = delimiter
        self.config    = config
        
    @property
    def value(self):
        """
        Return the parsed btye string as a DataFrame
        """
        try:
            return self._value
        except AttributeError:
            from io import BytesIO
            from dask.bytes.utils import read_block
            
            # read the relevant bytes
            with open(self.filename, 'rb') as f:
                block = read_block(f, self.offset, self.blocksize, self.delimiter)
                
            # parse the byte string
            b = BytesIO()
            b.write(block); b.seek(0)
            self._value = read_csv(b, **self.config)
                        
            return self._value

def make_partitions(filename, blocksize, config, delimiter="\n"):
    """
    Partition a CSV file into blocks, using the preferred blocksize 
    in bytes, returning the partititions and number of rows in 
    each partition

    This divides the input file into partitions with size
    roughly equal to blocksize, reads the bytes, and counts
    the number of delimiters to compute the size of each block
    
    Parameters
    ----------
    filename : str
        the name of the CSV file to load
    blocksize : int
        the desired number of bytes per block 
    delimiter : str, optional
        the character separating lines; default is
        the newline character
    config : dict
        any keyword options to pass to :func:`pandas.read_csv`
    
    Returns
    -------
    partitions : list of CSVPartition
        list of objects storing the data content of each file partition,
        stored as a bytestring
    sizes : list of int
        the list of the number of rows in each partition
    """
    from dask.bytes.utils import read_block
    
    config = config.copy()
    
    # search for lines separated by this character
    delimiter = delimiter.encode()

    # size in bytes and byte offsets of each partition
    size = os.path.getsize(filename)
    offsets = list(range(0, size, int(blocksize)))
    
    # skip blank lines
    skip_blank_lines = config.get('skip_blank_lines', True)
        
    # number of rows to read  
    nrows = config.pop('nrows', None)
        
    sizes = []; partitions = []
    with open(filename, 'rb') as f:
        for i, offset in enumerate(offsets):
            
            # skiprows only valid for first block
            if i > 0 and 'skiprows' in config:
                config.pop('skiprows')
                
            # set nrows for this block
            config['nrows'] = nrows
            
            block = read_block(f, offset, blocksize, delimiter)
            partitions.append(CSVPartition(filename, offset, blocksize, delimiter, **config))
            
            # count delimiter to get size
            size = block.count(delimiter)
                    
            # account for blank lines
            if skip_blank_lines:
                size -= block.count(delimiter+delimiter)
                if i == 0 and block.startswith(delimiter):
                    size -= 1
                    
            # account for skiprows
            skiprows = config.get('skiprows', 0)
            size -= skiprows
            
            # account for nrows
            if nrows is not None and nrows > 0:
                if nrows < size: 
                    sizes.append(nrows)
                    break
                else:
                    nrows -= size # update for next block
            sizes.append(size)
            
            
    
    return partitions, sizes

def verify_data(path, names, nrows=10, **config):
    """
    Verify the data by reading the first few lines of the specified 
    CSV file to determine the data type
    
    Parameters
    ----------
    path : str
        the name of the CSV file to load
    names : list of str
        the list of the names of the columns in the CSV file
    nrows : int, optional
        the number of rows to read from the file in order
        to infer the data type; default is 10
    **config : key, value pairs
        additional keywords to pass to :func:`pandas.read_csv`
    
    Returns
    -------
    dtype : dict
        dictionary holding the dtype for each name in `names`
    """    
    # read the first few lines to get the the dtype
    try:
        df = read_csv(path, nrows=nrows, names=names, **config)
        
        if df.isnull().sum().any():
            raise ValueError("'NaN' entries found when reading first %d rows; likely configuration error" %nrows)
        if any(dt == 'O' for dt in df.dtypes):
            raise ValueError("'object' data types found when reading first %d rows; likely configuration error" %nrows)
        
    except:
        import traceback
        config['names'] = names
        msg = ("error trying to read data with pandas.read_csv; ensure that 'names' matches "
               "the number of columns in the file and the file contains no comments\n")
        msg += "pandas configuration: %s\n" %str(config)
        msg += "\n%s" %(traceback.format_exc())
        raise ValueError(msg)

    toret = {}
    for name in df.columns:
        toret[name] = df[name].dtype
    return toret


class CSVFile(FileType):
    """
    A file object to handle the reading of columns of data from
    a CSV file

    Internally, this class partitions the CSV file into chunks, and 
    data is only read from the relevant chunks of the file, using
    :func:`pandas.read_csv`  
    
    This setup provides a significant speed-up when reading
    from the end of the file, since the entirety of the data
    does not need to be read first.
    
    The class supports any of the configuration keywords that can be 
    passed to :func:`pandas.read_csv`
    
    .. warning::
        This assumes the delimiter for separate lines is the newline
        character and that all columns in the file represent data 
        columns (no "index" column when using ``pandas``)
    """
    def __init__(self, path, names, blocksize=32*1024*1024, dtype={}, 
                    usecols=None, delim_whitespace=True, **config):
        """
        Parameters
        ----------
        path : str
            the name of the file to load
        names : list of str
            the names of the columns of the csv file; this should give
            names of all the columns in the file -- pass ``usecols``
            to select a subset of columns
        blocksize : int; optional
            the file will be partitioned into blocks of bytes roughly
            of this size
        dtype : dict, str; optional
            if specified as a string, assume all columns have this dtype,
            otherwise; each column can have a dtype entry in the dict;
            if not specified, the data types will be inferred from the file
        usecols : list; optional
            a ``pandas.read_csv``; a subset of ``names`` to store, ignoring
            all other columns
        delim_whitespace : bool; optional
            a ``pandas.read_csv`` keyword; if the CSV file is space-separated, 
            set this to ``True``
        config : 
            additional keyword pairs to pass to :func:`pandas.read_csv`; 
            see the documentation of that function for a full list
        """        
        self.path      = path
        self.names     = names if usecols is None else usecols
        self.blocksize = blocksize
        
        # ensure that no index column is passed
        if 'index_col' in config and config['index_col']:
            raise ValueError("'index_col = False' is not supported in CSVFile")
        config['index_col'] = False # no index columns in file
        
        # manually remove comments 
        if 'comment' in config and config['comment'] is not None:
            raise ValueError("please manually remove all comments from file")
        config['comment'] = None # no comments
        
        # ensure that no header is passed
        if 'header' in config and config['header'] is not None:
            raise ValueError("'header' not equal to None is not supported in CSVFile")
        config['header'] = None # no header
        
        if isinstance(config.get('skiprows', None), list):
            raise ValueError("only integer values supported for 'skiprows' in CSVFile")
                
        if 'skipfooter' in config:
            raise ValueError("'skipfooter' not supported in CSVFile")
                
        # set the read_csv defaults
        if 'sep' in config or 'delimiter' in config:
            delim_whitespace = False
        config['delim_whitespace'] = delim_whitespace
        config['usecols'] = usecols
        config.setdefault('engine', 'c')
        config.setdefault('skip_blank_lines', True)
        self.pandas_config = config.copy()
        
        # verify the data
        inferred_dtype = verify_data(self.path, names, **self.pandas_config)
        
        # dtype can also be a string --> apply to all columns
        if isinstance(dtype, string_types):
            dtype = {col:dtype for col in self.names}

        # store the dtype as a list
        self.dtype = []
        for col in self.names:
            if col in dtype:
                dt = dtype[col]
                if not isinstance(dt, numpy.dtype):
                    dt = numpy.dtype(dt)
            elif col in inferred_dtype:
                dt = inferred_dtype[col]
            else:
                raise ValueError("data type for column '%s' cannot be inferred from file" %col)
            self.dtype.append((col, dt))
        self.dtype = numpy.dtype(self.dtype)
        
        # add the dtype and names to the pandas config
        if config['engine'] == 'c':
            self.pandas_config['dtype'] = {col:self.dtype[col] for col in self.names}
        self.pandas_config['names'] = names

        # make the partitions
        self.partitions, self._sizes = make_partitions(path, blocksize, self.pandas_config)
        self.size = sum(self._sizes)

    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the file (in particles)
        
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
        numpy.array
            structured array holding the requested columns over
            the specified range of rows
        """
        if isinstance(columns, string_types): columns = [columns]
        
        toret = []
        for fnum in tools.get_file_slice(self._sizes, start, stop):

            # the local slice
            sl = tools.global_to_local_slice(self._sizes, start, stop, fnum)
            
            # access the dataframe of this partition
            data = self.partitions[fnum].value

            # slice and convert to a structured array
            data = data[sl[0]:sl[1]]
            data = data[columns]
            toret.append(data.to_records(index=False))
            
        return numpy.concatenate(toret, axis=0)[::step]
