from nbodykit.base.particles import ParticleSource
from nbodykit import CurrentMPIComm

import numpy

class UniformParticles(ParticleSource):
    @CurrentMPIComm.enable
    def __init__(self, nbar, BoxSize, seed=None, comm=None):
        self.comm    = comm

        _BoxSize = numpy.empty(3, dtype='f8')
        _BoxSize[:] = BoxSize
        self.attrs['BoxSize'] = _BoxSize

        rng = numpy.random.RandomState(seed)
        N = rng.poisson(nbar * numpy.prod(self.attrs['BoxSize']))
        seeds = rng.randint(0, high=comm.size, size=comm.size)
        rng = numpy.random.RandomState(seeds[comm.rank])

        start = N * comm.rank // comm.size
        end = N * (comm.rank  + 1)// comm.size

        self._size =  end - start
        self._pos = rng.uniform(size=(self._size, 3)) * self.attrs['BoxSize']
        self._vel = rng.uniform(size=(self._size, 3)) * self.attrs['BoxSize'] * 0.01

        ParticleSource.__init__(self, comm=comm)

    @property
    def size(self):
        return self._size

    @property
    def hcolumns(self):
        return ['Position', 'Velocity']

    def get_column(self, col):
        import dask.array as da
        if col == 'Position':
            r = da.from_array(self._pos, chunks=100000)
        if col == 'Velocity':
            r = da.from_array(self._vel, chunks=100000)
        r.attrs = {}
        return r


