import numpy

from ..extern.six import string_types
from . import FileType, tools

class CSVFile(FileType):
    """
    A file object to handle the reading of columns of data from
    a CSV file

    Internally, this class uses :func:`dask.dataframe.read_csv` 
    (which uses func:`pandas.read_csv`) to partition the CSV file
    into chunks, and data is only read from the relevant chunks 
    of the file. 
    
    This setup provides a significant speed-up when reading
    from the end of the file, since the entirety of the data
    does not need to be read first.
    
    The class supports any of the configuration keywords that can be 
    passed to :func:`pandas.read_csv`
    
    .. warning::

        This assumes the delimiter for separate lines is the newline
        character.
    """
    plugin_name = 'CSVFile'

    def __init__(self, path, names, blocksize=32*1024*1024, dtype={}, 
                    delim_whitespace=True, header=None, **config):
                    
        # used to read partitions of the file
        import dask.dataframe as dd
        
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
            inferred_dtype = tools.infer_csv_dtype(self.path, self.names, **self._config)

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
        self.partitions = dd.read_csv(self.path, **kws)
         
        # size is the sum of the size of each partition
        self._sizes = tools.csv_partition_sizes(self.path, self.blocksize)       
        self.size = sum(self._sizes)

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
        for fnum in tools.get_file_slice(self._sizes, start, stop):

            # the local slice
            sl = tools.global_to_local_slice(self._sizes, start, stop, fnum)
            
            # dataframe to structured array
            data = (self.partitions[fnum][sl[0]:sl[1]]).compute()
            data = data[columns]
            toret.append(data.to_records(index=False))
            
        return numpy.concatenate(toret, axis=0)[::step]
