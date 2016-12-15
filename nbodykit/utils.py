import contextlib
import numpy

@contextlib.contextmanager
def MPINumpyRNGContext(seed, comm):
    """
    A context manager (for use with the ``with`` statement) that will 
    seed the numpy random number generator (RNG) to a specific value,
    and then restore the RNG state back to whatever it was before.
    
    Notes
    -----
    This attempts to avoid correlation between random states for different 
    ranks by using the global seed to generate new seeds for each rank. 
    
    The seed must be a 32 bit unsigned integer, so it 
    is selected between 0 and 4294967295
    
    Parameters
    ----------
    seed : int, None
        the global seed, used to seed the local random state
    comm : MPI.Communicator
        the MPI communicator
    
    Yields
    ------
    int : 
        the integer used to seed the random state on the local rank
    """ 
    from astropy.utils.misc import NumpyRNGContext
    
    try:
        # create a global random state
        rng = numpy.random.RandomState(seed)
    
        # use the global seed to seed all ranks
        # seed must be an unsigned 32 bit integer (0xffffffff in hex)
        seeds = rng.randint(0, 4294967295, size=comm.size)
    
        # choose the right local seed for this rank
        local_seed = seeds[comm.rank]
    
        with NumpyRNGContext(local_seed):
            yield local_seed
    except:
        pass


def cosmology_to_dict(cosmo, prefix='cosmo.'):
    try: import classylss
    except: raise ImportError("`classylss` is required to use %s" %self.__class__.__name__)
    pars = classylss.ClassParams.from_astropy(cosmo)

    d = {}
    for key, value in pars.items():
        try: 
            value = float(value)
        except ValueError:
            pass
        d[prefix + key] = value
    return d

