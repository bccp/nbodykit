import psutil
from mpi4py import MPI

def auto_blocksize(total_memory, cpu_count):
    memory_factor = 10
    blocksize = int(total_memory // cpu_count / memory_factor)
    return min(blocksize, int(64e6))

TOTAL_MEM = psutil.virtual_memory().total
CPU_COUNT = psutil.cpu_count()
AUTO_BLOCKSIZE = auto_blocksize(TOTAL_MEM, CPU_COUNT)

def get_local_slice(total_size, N, i):
    """
    Partition an object of size `total_size` into `N` partitions, 
    and return the (start, stop) index bounds for the partition
    with number `i`
    
    Parameters
    ----------
    total_size : int
        the total size of the object we are partitioning
    N : int
        the number of partitions
    i : int
        the index of the partition bounds to return
    
    Returns
    -------
    start, stop : int
        the indices inidicating the beginning and end of the partition
    """
    Neach_part, extras = divmod(total_size, N)
    part_sizes = numpy.zeros(N+1, dtype='i8')
    part_sizes[1:] = numpy.cumsum(extras * [Neach_part+1] + (N-extras) * [Neach_part])
    return part_sizes[i], part_sizes[i+1]

def FileIterator(f, columns, chunksize=None, comm=None):
    """
    Iterate through a file
    """
    # get default chunksize based on memory and itemsize
    if chunksize is None:
        itemsize = sum(f.dtype[col].itemsize for col in columns)
        chunksize = AUTO_BLOCKSIZE // itemsize
        
    # the comm
    if comm is None: comm = MPI.COMM_WORLD
    
    # get the local partition and its size
    start, stop = get_local_slice(f.size, comm.size, comm.rank)
    
    # yield the local file chunks
    for chunk in f.get_partition(columns, start, stop, chunksize=chunksize):
        yield chunk


 

        
        
        
    
        
        
    
    