from __future__ import print_function

import numpy
import logging
from nbodykit.base.particles import ParticleSource

from mpi4py import MPI
import warnings
from pmesh.pm import RealField

def removeradiation(cosmo):
    if cosmo.Ogamma0 != 0:
        warnings.warn("Cosmology has radiation. Radiation is removed from background for a consistent integration.")

    return cosmo.clone(Tcmb0=0)

def za_transfer(deltain, deltaout, dir):
    for k, i, slabin, slabout in zip(deltain.slabs.x,
                    deltain.slabs.i, deltain.slabs, deltaout.slabs):
        kk = sum(ki ** 2 for ki in k)
        mask = numpy.ones(slabin.shape, '?')

        for ii, n in zip(i, deltain.Nmesh):
           mask &=  ii >= n // 2
        mask[kk == 0] = True

        kk[kk == 0] = 1
        slabout[...] = (~mask) * slabin * 1j * k[dir] / kk

class LPTParticles(ParticleSource):
    logger = logging.getLogger('LPT')

    def __init__(self, complex, cosmo, redshift=1.0):
        comm = complex.pm.comm

        cosmo = removeradiation(cosmo)
        self.cosmo = cosmo

        self.attrs.update(complex.attrs)
        self.attrs['redshift'] = redshift

        self._source = {}
        self._source['InitialPosition'] = self._fill_initial_position(complex)
        self._source['dx1'] = self._fill_dx1(complex)

        ParticleSource.__init__(self, comm=comm)
        D1, f1, D2, f2 = cosmo.lptode(z=redshift)
        a = 1 / (redshift + 1.)
        E = cosmo.efunc(z=redshift)

        self._fallbacks['Position'] = self['InitialPosition'] + self['dx1'] * D1
        self._fallbacks['Velocity'] = self['dx1'] * D1 * f1 * a ** 2 * E

    @property
    def size(self):
        return len(self._source['InitialPosition'])

    def get_column(self, col):
        import dask.array as da
        return da.from_array(self._source[col], chunks=100000)

    @property
    def hcolumns(self):
        return list(self._source.keys())

    def _fill_initial_position(self, complex):
        basepm = complex.pm
        ndim = len(basepm.Nmesh)
        real = RealField(basepm)

        dtype = numpy.dtype(('f4', 3))

        # one particle per base mesh point
        source = numpy.zeros((real.size, ndim), dtype='f4')

        for d in range(len(real.shape)):
            real[...] = 0
            for xi, slab in zip(real.slabs.i, real.slabs):
                slab[...] = xi[d] * (real.BoxSize[d] / real.Nmesh[d])
            source[..., d] = real.value.flat
        return source

    def _fill_dx1(self, complex):
        basepm = complex.pm
        ndim = len(basepm.Nmesh)

        source = numpy.zeros((self.size, ndim), dtype='f4')
        for d in range(len(basepm.Nmesh)):
            delta_k = complex.copy()
            za_transfer(complex, delta_k, d)
            disp = delta_k.c2r(delta_k)
            source[..., d] = disp.value.flat

        return source
