from __future__ import print_function

import numpy
import logging
from nbodykit.base.particles import ParticleSource

from mpi4py import MPI
import warnings

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

    def __init__(self, dlink, cosmo, redshift=0.0):
        comm = dlink.pm.comm

        cosmo = removeradiation(cosmo)
        self.cosmo = cosmo

        if hasattr(dlink, 'attrs'):
            self.attrs.update(dlink.attrs)

        self.attrs['redshift'] = redshift
        self.basepm = dlink.pm
        self._source = {}
        self._source['InitialPosition'] = self._compute_initial_position(self.basepm)
        self._source['LPTDisp1'] = self._compute_dx1(dlink.value, self.basepm)

        ParticleSource.__init__(self, comm=comm)

        import dask.array as da
        self._fallbacks['GradPosition'] = da.zeros((self.size, 3), dtype='f4', chunks=100000)
        self._fallbacks['GradVelocity'] = da.zeros((self.size, 3), dtype='f4', chunks=100000)
        self.set_redshift(redshift)

    def set_redshift(self, redshift):
        self.redshift = redshift

    def gradient(self):
        return self._grad_dx1(self.basepm)

    @property
    def size(self):
        return len(self._source['InitialPosition'])

    def get_column(self, col):
        import dask.array as da
        if col in self._source:
            return da.from_array(self._source[col], chunks=100000)

        redshift = self.redshift
        cosmo = self.cosmo

        D1, f1, D2, f2 = cosmo.lptode(z=redshift)
        a = 1 / (redshift + 1.)
        E = cosmo.efunc(z=redshift)

        d = {}
        d['Position'] = self['InitialPosition'] + self['LPTDisp1'] * D1
        d['Velocity'] = self['LPTDisp1'] * D1 * f1 * a ** 2 * E
        d['GradLPTDisp1'] = self['GradVelocity'] * (D1 * f1 * a **2 * E) + self['GradPosition'] * D1

        return d[col]

    @property
    def hcolumns(self):
        return list(self._source.keys()) + ['Position', 'Velocity', 'GradLPTDisp1']

    def _compute_initial_position(self, basepm):
        ndim = len(basepm.Nmesh)
        real = basepm.create('real')

        dtype = numpy.dtype(('f4', 3))

        # one particle per base mesh point
        source = numpy.zeros((real.size, ndim), dtype='f4')

        for d in range(len(real.shape)):
            real[...] = 0
            for xi, slab in zip(real.slabs.i, real.slabs):
                slab[...] = xi[d] * (real.BoxSize[d] / real.Nmesh[d])
            source[..., d] = real.value.flat
        return source

    def _compute_dx1(self, dlink, basepm):
        ndim = len(basepm.Nmesh)

        source = numpy.zeros((self.size, ndim), dtype='f4')
        delta_k = basepm.create('complex')
        for d in range(len(basepm.Nmesh)):
            delta_k[...] = dlink
            za_transfer(delta_k, delta_k, d)
            disp = delta_k.c2r(delta_k)
            source[..., d] = disp.value.flat

        return source

    def _grad_dx1(self, basepm):
        ndim = len(basepm.Nmesh)
        grad = basepm.create('complex')
        grad[...] = 0
        source = self['GradLPTDisp1']
        print(self['GradPosition'].compute())
        for d in range(len(basepm.Nmesh)):
            grad_disp = basepm.create('real')
            grad_disp.value.flat[...] = source[..., d].compute()
            grad_disp_k = grad_disp.c2r_gradient(grad_disp)
            za_transfer(grad_disp_k, grad_disp_k, d)
            grad.value[...] += grad_disp_k.value

        return grad
