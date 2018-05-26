from nbodykit.base.catalog import CatalogSource, column
from nbodykit import CurrentMPIComm
from nbodykit.mpirng import MPIRandomState
import numpy

class RandomCatalog(CatalogSource):
    """
    A CatalogSource that can have columns added via a
    collective random number generator.

    The random number generator stored as :attr:`rng` behaves
    as :class:`numpy.random.RandomState` but generates random
    numbers only on the local rank in a manner independent of
    the number of ranks.

    Parameters
    ----------
    csize : int
        the desired collective size of the Source
    seed : int, optional
        the global seed for the random number generator
    comm : MPI communicator
        the MPI communicator; set automatically if None
    """
    def __repr__(self):
        args = (self.size, self.attrs['seed'])
        return "RandomCatalog(size=%d, seed=%s)" % args

    @CurrentMPIComm.enable
    def __init__(self, csize, seed=None, comm=None):

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
        start = comm.rank * csize // comm.size
        end   = (comm.rank + 1) * csize // comm.size
        self._size =  end - start

        self._rng = MPIRandomState(comm, seed=seed, size=self._size)

        # init the base class
        CatalogSource.__init__(self, comm=comm)

    @property
    def rng(self):
        """
        A :class:`MPIRandomState` that behaves as
        :class:`numpy.random.RandomState` but generates random
        numbers in a manner independent of the number of ranks.
        """
        return self._rng

class UniformCatalog(RandomCatalog):
    """
    A CatalogSource that has uniformly-distributed ``Position``
    and ``Velocity`` columns.

    The random numbers generated do not depend on the number of
    available ranks.

    Parameters
    ----------
    nbar : float
        the desired number density of particles in the box
    BoxSize : float, 3-vector
        the size of the box
    seed : int, optional
        the random seed
    comm :
        the MPI communicator
    """
    def __repr__(self):
        args = (self.size, self.attrs['seed'])
        return "UniformCatalog(size=%d, seed=%s)" % args

    @CurrentMPIComm.enable
    def __init__(self, nbar, BoxSize, seed=None, dtype='f8', comm=None):

        self.comm    = comm

        _BoxSize = numpy.empty(3, dtype='f8')
        _BoxSize[:] = BoxSize
        self.attrs['BoxSize'] = _BoxSize

        rng = numpy.random.RandomState(seed)
        N = rng.poisson(nbar * numpy.prod(self.attrs['BoxSize']))
        if N == 0:
            raise ValueError("no uniform particles generated, try increasing `nbar` parameter")
        RandomCatalog.__init__(self, N, seed=seed, comm=comm)

        self._pos = (self.rng.uniform(itemshape=(3,)) * self.attrs['BoxSize']).astype(dtype)
        self._vel = (self.rng.uniform(itemshape=(3,)) * self.attrs['BoxSize'] * 0.01).astype(dtype)

    @column
    def Position(self):
        """
        The position of particles, uniformly distributed in :attr:`BoxSize`
        """
        return self.make_column(self._pos)

    @column
    def Velocity(self):
        """
        The velocity of particles, uniformly distributed in ``0.01 x BoxSize``
        """
        return self.make_column(self._vel)
