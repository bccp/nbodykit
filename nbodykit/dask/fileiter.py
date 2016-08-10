import psutil
from mpi4py import MPI

def auto_blocksize(total_memory, cpu_count):
    memory_factor = 10
    blocksize = int(total_memory // cpu_count / memory_factor)
    return min(blocksize, int(64e6))

TOTAL_MEM = psutil.virtual_memory().total
CPU_COUNT = psutil.cpu_count()
AUTO_BLOCKSIZE = auto_blocksize(TOTAL_MEM, CPU_COUNT)

def get_Nlocal(N, size, rank):
    """
    Return the local number of objects on each rank, given
    the desired total `N`, the communicator `size`, and the
    `rank`
    """
    Neach_section, extras = divmod(N, size)
    section_sizes = extras * [Neach_section+1] + (size-extras) * [Neach_section]
    return section_sizes[rank]
     
def FileIterator(f, columns, chunksize=None, comm=None):
    """
    Iterate through a file
    """
    # get default chunksize based on memory and itemsize
    if chunksize is None:
        chunksize = AUTO_BLOCKSIZE // f.dtype.itemsize
        
    # the comm
    if comm is None: comm = MPI.COMM_WORLD
        
    # get the local partition and its size
    partition = f.partition(columns, comm.size)[comm.rank]
    N = len(partition)
    
    # yield chunks of the local partition in chunksize units
    start = 0; stop = chunksize
    while start < N:
        yield partition[start:stop]     
        start = stop
        stop = min(start+chunksize, N)
        
        
        
    
        
        
    
    