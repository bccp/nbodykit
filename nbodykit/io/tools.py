import numpy
import os

def get_slice_size(start, stop, step):
    """
    Utility function to return the size of an array slice
    
    Parameters
    ----------
    start : int
        the beginning of the slice
    stop : int
        the end of the slice
    step : int
        the slice step size
    
    Returns
    -------
    N : int
        the total size of the slice
    """
    N, remainder = divmod(stop-start, step)
    if remainder: N += 1
    return N

def csv_partition_sizes(filename, blocksize, delimiter="\n"):
    """
    From a filename and preferred blocksize in bytes,
    return the number of rows in each partition

    This divides the input file into partitions with size
    roughly equal to blocksize, reads the bytes, and counts
    the number of delimiters
    
    Parameters
    ----------
    filename : str
        the name of the CSV file to load
    blocksize : int
        the desired number of bytes per block 
    delimiter : str, optional
        the character separating lines; default is
        the newline character
    
    Returns
    -------
    nrows : list of int
        the list of the number of rows in each block
    """
    from dask.bytes.core import read_block_from_file

    # search for lines separated by newline char
    delimiter = delimiter.encode()

    # size in bytes and byte offsets of each partition
    size = os.path.getsize(filename)
    offsets = list(range(0, size, int(blocksize)))

    nrows = []
    for offset in offsets:
        block = read_block_from_file(open(filename, 'rb'), offset, blocksize, delimiter)
        nrows.append(block.count(delimiter))
    return nrows

def infer_csv_dtype(path, names, nrows=10, **config):
    """
    Read the first few lines of the specified CSV file to determine 
    the data type
    
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
    from pandas import read_csv 
    
    # read the first few lines to get the the dtype
    df = read_csv(path, nrows=nrows, names=names, **config)

    toret = {}
    for name in names:
        toret[name] = df[name].dtype
    return toret
    

def global_to_local_slice(sizes, start, stop, fnum):
    """
    Convert a global slice, specified by `start` and `stop` 
    to the corresponding local indices of the file specified
    by `fnum`
    
    Parameters
    ----------
    sizes : array_like
        the sizes of each file in the file stack
    start : int
        the global index to begin the slice
    stop : int
        the global index to stop the slice
    fnum : int
        the file number that defines the desired local indexing
    
    Returns
    -------
    local_start, local_stop : int
        the local start and stop slice values
    """
    # cumulative sizes
    cumsizes = numpy.insert(numpy.cumsum(sizes), 0, 0)
    
    # local slice
    start_size = cumsizes[fnum] 
    return (max(start-start_size, 0), min(stop-start_size, sizes[fnum]))

def get_file_slice(sizes, start, stop):
    """
    Return the list of file numbers that must be accessed
    to return data between `start` and `slice`, where
    these indices are defined in terms of the global
    catalog indexing
    
    Parameters
    ----------
    sizes : array_like
        the sizes of each file in the file stack
    start : int
        the global index to begin the slice
    stop : int
        the global index to stop the slice
    
    Returns
    -------
    fnums : list
        the list of integers specifying the relevant
        file numbers that must be accessed
    """
    # cumulative sizes
    cumsizes = numpy.insert(numpy.cumsum(sizes), 0, 0)
    
    # return the relevant file numbers
    fnums = numpy.searchsorted(cumsizes[1:], [start, stop])
    return list(range(fnums[0], fnums[1]+1))
