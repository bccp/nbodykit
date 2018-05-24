from numpy.random import RandomState
import numpy
import functools
import contextlib

N_PER_SEED = 100000

def _mpi_enabled_rng(func):
    """
    A decorator that handles generating random numbers in parallel
    in a manner that is independent of the number of ranks

    Designed to be used with :class:`MPIRandomState`
    """
    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):

        self = func.__self__
        size = kwargs.get('size', None)
        if size is not None and isinstance(size, int):
            size = (size,)

        # do nothing if size not provided or wrong
        if size is None or size[0] != self.size:
            return func(*args, **kwargs)

        # size matches the "size" attribute
        else:
            toret = []

            # loop through chunks that this rank is responsible for
            for chunk, (start, stop) in zip(self._chunks, self._slices):
                with self.seeded_context(self._seeds[chunk]):

                    kwargs['size'] = (N_PER_SEED,) + size[1:]
                    if chunk == self.N // N_PER_SEED:
                        kwargs['size'] = (self.N % N_PER_SEED,) + size[1:]
                    toret.append(func(*args, **kwargs)[start:stop])
            return numpy.concatenate(toret, axis=0)

    func_wrapper.mpi_enabled = True
    return func_wrapper

class MPIRandomState(RandomState):
    """
    A wrapper around :class:`numpy.random.RandomState` that can return
    random numbers in parallel, independent of the number of ranks.

    Parameters
    ----------
    comm : MPI communicator
        the MPI communicator
    seed : int
        the global seed that seeds all other random seeds
    localsize : int
        the local size of the random numbers to generate; we return chunks of
        the total on each CPU, based on the CPU's rank
    """
    def __init__(self, comm, seed, localsize):

        RandomState.__init__(self, seed=seed)

        N = comm.allreduce(numpy.int64(localsize))

        # the number of seeds to generate N particles
        n_seeds = N // N_PER_SEED
        if N % N_PER_SEED: n_seeds += 1

        self.comm        = comm
        self.global_seed = seed
        self.N           = N

        start = numpy.sum(comm.allgather(localsize)[:comm.rank], dtype='intp')
        stop  = start + localsize
        self.size  = stop - start

        # generate the full set of seeds from the global seed
        rng = numpy.random.RandomState(seed=seed)
        self._seeds = rng.randint(0, high=0xffffffff, size=n_seeds)

        # sizes of each chunk
        sizes = [N_PER_SEED]*(N//N_PER_SEED)
        if N % N_PER_SEED: sizes.append(N % N_PER_SEED)

        # the local chunks this rank is responsible for
        cumsizes = numpy.insert(numpy.cumsum(sizes), 0, 0)
        chunk_range = numpy.searchsorted(cumsizes[1:], [start, stop])
        self._chunks = list(range(chunk_range[0], chunk_range[1]+1))

        # and the slices for each local chunk
        self._slices = []
        for chunk in self._chunks:
            start_size = cumsizes[chunk]
            sl = (max(start-start_size, 0), min(stop-start_size, sizes[chunk]))
            self._slices.append(sl)

    def __dir__(self):
        """
        Explicitly set the attributes as those of the RandomState class too
        """
        d1 = set(RandomState().__dir__())
        d2 = set(RandomState.__dir__(self))
        return list(d1|d2)

    def __getattribute__(self, name):
        """
        Decorate callable functions of RandomState such that they return
        chunks of the total `N` random numbers generated
        """
        attr = RandomState.__getattribute__(self, name)
        if callable(attr) and not getattr(attr, 'mpi_enabled', False):
            attr = _mpi_enabled_rng(attr)
        return attr

    @contextlib.contextmanager
    def seeded_context(self, seed):
        """
        A context manager to set and then restore the random seed
        """
        startstate = self.get_state()
        self.seed(seed)
        yield
        self.set_state(startstate)

