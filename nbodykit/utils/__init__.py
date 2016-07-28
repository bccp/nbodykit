"""
General utility functions
"""
import numpy

def local_random_seed(seed, comm):
    """
    Return a random seed for the local rank, 
    which is seeded by the global `seed`
    
    Notes
    -----
    This attempts to avoid correlation between random
    states for different ranks by using the global seed
    to generate new seeds for each rank. 
    
    The seed must be a 32 bit unsigned integer, so it 
    is selected between 0 and 4294967295
    
    Parameters
    ----------
    seed : int, None
        the global seed, used to seed the local random state
    comm : MPI.Communicator
        the MPI communicator
    
    Returns
    -------
    int : 
        a integer appropriate for seeding the local random state
    """ 
    # create a global random state
    rng = numpy.random.RandomState(seed)
    
    # use the global seed to seed all ranks
    # seed must be an unsigned 32 bit integer (0xffffffff in hex)
    seeds = rng.randint(0, 4294967295, size=comm.size)
    
    # choose the right local seed for this rank
    return seeds[comm.rank]
    
def timer(start, end):
    """
    Utility function to return a string representing the elapsed time, 
    as computed from the input start and end times
    
    Parameters
    ----------
    start : int
        the start time in seconds
    end : int
        the end time in seconds
    
    Returns
    -------
    str : 
        the elapsed time as a string, using the format `hours:minutes:seconds`
    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
