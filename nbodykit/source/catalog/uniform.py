from nbodykit.base.catalog import CatalogSource, column
from nbodykit import CurrentMPIComm

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
    Wrapper around :class:`numpy.random.RandomState` that can return
    random numbers in parallel, independent of the number of ranks
    """
    def __init__(self, comm, seed, N):
        """
        Parameters
        ----------
        comm : MPI communicator
            the MPI communicator
        seed : int
            the global seed that seeds all other random seeds
        N : int
            the total size of the random numbers to generate; we return chunks of
            the total on each CPU, based on the CPU's rank
        """
        RandomState.__init__(self, seed=seed)

        # the number of seeds to generate N particles
        n_seeds = N // N_PER_SEED
        if N % N_PER_SEED: n_seeds += 1

        self.comm        = comm
        self.global_seed = seed
        self.N           = N

        start = N * comm.rank // comm.size
        stop  = N * (comm.rank  + 1) // comm.size
        self.size  = stop - start

        # generate the full set of seeds from the global seed
        self._seeds = self.randint(0, high=0xffffffff, size=n_seeds)

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

class RandomCatalog(CatalogSource):
    """
    A catalog source that can have columns added via a
    collective random number generator

    The random number generator stored as :attr:`rng` behaves
    as :class:`numpy.random.RandomState` but generates random
    numbers only on the local rank in a manner independent of
    the number of ranks
    """
    def __repr__(self):
        return "RandomCatalog(seed=%(seed)s)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self, csize, seed=None, comm=None, use_cache=False):
        """
        Parameters
        ----------
        csize : int
            the desired collective size of the Source
        seed : int; optional
            the global seed for the random number generator
        comm : MPI communicator
            the MPI communicator; set automatically if None
        """
        self.comm = comm

        # set the seed randomly if it is None
        if seed is None:
            if self.comm.rank == 0:
                seed = numpy.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs['seed'] = seed

        # generate the seeds from the global seed
        if csize == 0:
            raise ValueError("no random particles generated!")
        self.rng = MPIRandomState(comm, seed, csize)
        self._size =  self.rng.size

        # init the base class
        CatalogSource.__init__(self, comm=comm, use_cache=use_cache)

    @property
    def size(self):
        return self._size


class UniformCatalog(RandomCatalog):
    """
    A catalog source that has uniformly-distributed ``Position``
    and ``Velocity`` columns

    The random numbers generated do not depend on the number of
    ranks, i.e., ``comm.size``.
    """
    def __repr__(self):
        return "UniformCatalog(seed=%(seed)s)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self, nbar, BoxSize, seed=None, comm=None, use_cache=False):
        """
        Parameters
        ----------
        nbar : float
            the desired number density of particles in the box
        BoxSize : float, 3-vector
            the size of the box
        seed : int; optional
            the random seed
        comm :
            the MPI communicator
        """
        self.comm    = comm

        _BoxSize = numpy.empty(3, dtype='f8')
        _BoxSize[:] = BoxSize
        self.attrs['BoxSize'] = _BoxSize

        rng = numpy.random.RandomState(seed)
        N = rng.poisson(nbar * numpy.prod(self.attrs['BoxSize']))
        if N == 0:
            raise ValueError("no uniform particles generated, try increasing `nbar` parameter")
        RandomCatalog.__init__(self, N, seed=seed, comm=comm, use_cache=use_cache)

        self._pos = self.rng.uniform(size=(self._size, 3)) * self.attrs['BoxSize']
        self._vel = self.rng.uniform(size=(self._size, 3)) * self.attrs['BoxSize'] * 0.01

    @column
    def Position(self):
        return self.make_column(self._pos)

    @column
    def Velocity(self):
        return self.make_column(self._vel)
