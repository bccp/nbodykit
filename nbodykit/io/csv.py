import pandas as pd
import os
import dask.dataframe as dd
import numpy

from ..extern.six import string_types
from . import FileType
from .stack import FileStack

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


class CSVFile(FileType):
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
        self.path      = path
        self.names     = names
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
            dtype = {col:dtype for col in self.names}

        # infer the data type?
        if not all(col in dtype for col in self.names):
            inferred_dtype = infer_dtype(self.path, self.names, self._config)

        # store the dtype as a list
        self.dtype = []
        for col in self.names:
            if col in dtype:
                dt = dtype[col]
                if not isinstance(dt, numpy.dtype):
                    dt = numpy.dtype(dt)
            else:
                dt = inferred_dtype[col]
            self.dtype.append((col, dt))
        self.dtype = numpy.dtype(self.dtype)
        
        # initialize the underlying dask partitions
        kws = self._config.copy()
        kws['dtype'] = {col:self.dtype[col] for col in self.names}
        kws['names'] = names
        kws['blocksize'] = self.blocksize
        kws['collection'] = False
        files = dd.read_csv(self.path, **kws)
        sizes = get_partition_sizes(self.path, self.blocksize)        
        self.stack = FileStack.from_files(files, sizes=sizes)
        
        # size is the sum of the size of each partition
        self.size = sum(self.stack.sizes)

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "a csv file reader"
        
        s.add_argument("path", type=str, 
            help='the name of the file to load')
        s.add_argument("names", nargs='+', type=str, 
            help='the names of each column in the csv file')
        s.add_argument("blocksize", type=int,
            help='internally partition the CSV file into blocks of this size (in bytes)')
        s.add_argument("dtype", 
            help=("a dictionary providing data types for the various columns; "
                    "data types not provided are inferred from the file"))
        s.add_argument("delim_whitespace", type=bool,
            help='set to True if the input file is space-separated')
        s.add_argument("header",
            help="the type of header in the CSV file; if no header, set to None")

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
        
        toret = []
        for fnum in self.stack._file_range(start, stop):

            # the local slice
            sl = self.stack._normalized_slice(start, stop, fnum)
            
            # dataframe to structured array
            data = (self.stack.files[fnum][sl[0]:sl[1]]).compute()
            data = data[columns]
            toret.append(data.to_records(index=False))
            
        return numpy.concatenate(toret, axis=0)[::step]
