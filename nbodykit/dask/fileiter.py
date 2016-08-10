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
    
    FIXME: no load balancing here -- only works well
    when the file size is much larger
    """
    # get default chunksize based on memory and itemsize
    if chunksize is None:
        itemsize = sum(f.dtype[col].itemsize for col in columns)
        chunksize = AUTO_BLOCKSIZE // itemsize
        
    # the comm
    if comm is None: comm = MPI.COMM_WORLD
    
    # get the local partition and its size
    partition = f.partition(columns, comm.size, chunksize=chunksize)[comm.rank]
    
    # yield the chunks of this partition
    start = stop = 0
    for size in partition.chunks[0]:
        start = stop
        stop += size
        yield partition[start:stop]

 

        
        
        
    
        
        
    
    