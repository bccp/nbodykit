import numpy

def consecutive_view_slices(start, stop, step, sizes):
    
    if isinstance(sizes, list):
        sizes = numpy.array(sizes, dtype='i8')
    total_size = sizes.sum()
    nfiles     = len(sizes)
    
    if start < 0: start += total_size
    if stop < 0: stop += total_size
    
    # determine the file numbers of start/stop from cumulative sizes
    cumsum = numpy.zeros(nfiles+1, dtype=sizes.dtype)
    cumsum[1:] = sizes.cumsum()
    fnums = numpy.searchsorted(cumsum[1:], [start, stop])
            
    # loop over necessary files to get requested range
    global_start = 0
    for fnum in range(fnums[0], fnums[1]+1):
                    
        # normalize the global start/stop to the per file values     
        start_size = cumsum[fnum] 
        this_slice = (max(start-start_size, 0), min(stop-start_size, sizes[fnum]), step)

        yield fnum, this_slice