import numpy

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