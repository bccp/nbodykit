from nbodykit.base.particles import ParticleSource, column
from nbodykit import CurrentMPIComm

import numpy

class RandomParticles(ParticleSource):

    def __repr__(self):
        return "RandomParticles(seed=%(seed)s)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self, Ntot, seed=None, comm=None):
        
        self.comm    = comm
        self.attrs['seed'] = seed

        rng = numpy.random.RandomState(seed)
        seeds = rng.randint(0, high=comm.size, size=comm.size)
        self.rng = numpy.random.RandomState(seeds[comm.rank])

        start = Ntot * comm.rank // comm.size
        end = Ntot * (comm.rank  + 1)// comm.size

        self._size =  end - start
        ParticleSource.__init__(self, comm=comm)

        
    @property
    def size(self):
        return self._size


class UniformParticles(RandomParticles):

    def __repr__(self):
        return "UniformParticles(seed=%(seed)s)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self, nbar, BoxSize, seed=None, comm=None):
        self.comm    = comm

        _BoxSize = numpy.empty(3, dtype='f8')
        _BoxSize[:] = BoxSize
        self.attrs['BoxSize'] = _BoxSize

        rng = numpy.random.RandomState(seed)
        N = rng.poisson(nbar * numpy.prod(self.attrs['BoxSize']))
        RandomParticles.__init__(self, N, seed=seed, comm=comm)
        
        self._pos = rng.uniform(size=(self._size, 3)) * self.attrs['BoxSize']
        self._vel = rng.uniform(size=(self._size, 3)) * self.attrs['BoxSize'] * 0.01
        
    @column
    def Position(self):
        return self.make_column(self._pos)

    @column
    def Velocity(self):
        return self.make_column(self._vel)
