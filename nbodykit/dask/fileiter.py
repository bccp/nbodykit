import psutil
from mpi4py import MPI

def auto_blocksize(total_memory, cpu_count):
    memory_factor = 10
    blocksize = int(total_memory // cpu_count / memory_factor)
    return min(blocksize, int(64e6))

TOTAL_MEM = psutil.virtual_memory().total
CPU_COUNT = psutil.cpu_count()
AUTO_BLOCKSIZE = auto_blocksize(TOTAL_MEM, CPU_COUNT)
     
def FileIterator(f, columns, chunksize=None, comm=None):
    """
    Iterate through a file
    """
    # get default chunksize based on memory and itemsize
    if chunksize is None:
        chunksize = AUTO_BLOCKSIZE // f.dtype.itemsize
        
    # the comm
    if comm is None: comm = MPI.COMM_WORLD
        
    # partitions
    partitions = f.partition(columns, chunksize)

    # yield the paritions across all ranks
    for i in range(comm.rank, len(partitions), comm.size):
        yield partition[i]     

        
        
        
    
        
        
    
    