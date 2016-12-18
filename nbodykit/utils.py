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

def attrs_to_dict(obj, prefix):
    if not hasattr(obj, 'attrs'):
        return {}

    d = {}
    for key, value in obj.attrs.items():
        d[prefix + key] = value
    return d

import json
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {'__complex__': [obj.real, obj.imag ]}

        elif isinstance(obj, numpy.ndarray):
            value = obj
            dtype = obj.dtype
            d = {
                '__dtype__' :
                    dtype.str if dtype.names is None else dtype.descr,
                '__data__': value.tolist()
            }
            return d
        return json.JSONEncoder.default(self, obj)

class JSONDecoder(json.JSONDecoder):
    @staticmethod
    def hook(value):
        if '__dtype__' in value:
            dtype = value['__dtype__']

            if isinstance(dtype, list):
                true_dtype = []
                true_a = []
                for field in dtype:
                    if len(field) == 3:
                        true_dtype.append((str(field[0]), str(field[1]), field[2]))
                    if len(field) == 2:
                        true_dtype.append((str(field[0]), str(field[1])))
                a = [tuple(i) for i in value['__data__']]
            else:
                true_dtype = dtype
                a = value['__data__']
            return numpy.array(a, dtype=true_dtype)

        if '__complex__' in value:
            real, imag = value['__complex__']
            return real + 1j * imag

        return value

    def __init__(self, *args, **kwargs):
        kwargs['object_hook'] = JSONDecoder.hook
        json.JSONDecoder.__init__(self, *args, **kwargs)
