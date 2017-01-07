from __future__ import print_function

import numpy
import logging
from nbodykit.base.particles import ParticleSource, column
from nbodykit.cosmology import PerturbationGrowth

from mpi4py import MPI
import warnings

from nbodykit import fastpm

def removeradiation(cosmo):
    if cosmo.Ogamma0 != 0:
        warnings.warn("Cosmology has radiation. Radiation is removed from background for a consistent integration.")

    return cosmo.clone(Tcmb0=0)

class LPTParticles(ParticleSource):
    logger = logging.getLogger('LPT')

    def __init__(self, dlink, cosmo, redshift=0.0, order=2):
        comm = dlink.pm.comm

        cosmo = removeradiation(cosmo)
        self.cosmo = cosmo
        self.pt = PerturbationGrowth(cosmo)

        if hasattr(dlink, 'attrs'):
            self.attrs.update(dlink.attrs)

        self.attrs['redshift'] = redshift
        self.attrs['order'] = order

        self.basepm = dlink.pm
        self.dlink = dlink

        self._source = {}
        self._source['InitialPosition'] = fastpm.create_grid(self.basepm, shift=0.0)
        self._source['LPTDisp1'] = fastpm.lpt1(dlink, self._source['InitialPosition'])

        lpt2source = fastpm.lpt2source(dlink)

        self._source['LPTDisp2'] = fastpm.lpt1(lpt2source, self._source['InitialPosition'])

        ParticleSource.__init__(self, comm=comm)

        import dask.array as da
        self._fallbacks['GradPosition'] = da.zeros((self.size, 3), dtype='f4', chunks=100000)
        self._fallbacks['GradVelocity'] = da.zeros((self.size, 3), dtype='f4', chunks=100000)
        self.set_redshift(redshift)

    def set_redshift(self, redshift):
        self.redshift = redshift
        # now deal with generated sources
        cosmo = self.cosmo
        a = 1 / (1. + redshift)

        self.D1 = self.pt.D1(a)
        self.f1 = self.pt.f1(a)
        self.D2 = self.pt.D2(a)
        self.f2 = self.pt.f2(a)

        if self.attrs['order'] == 1:
            self.D2 = 0
            self.f2 = 0

        self.a = a
        self.E = self.pt.E(a)

    @staticmethod
    def gradient(dlink, self):
        """ Backtrace the gradient of LPT source at dlink, using
            self['GradLPTDisp1'] as the gradient column of LPT1 displacement
        """
        # path through first order LPT
        gradient = fastpm.lpt1_gradient(self.dlink, self._source['InitialPosition'], self['GradLPTDisp1'].compute())

        # path through second order order LPT
        # forward
        # because the exact value of lpt2source is irrelevant, we save some computation
        # by not using lpt2source = fastpm.lpt2source(self.dlink)

        lpt2source = self.dlink

        # backward
        gradient_lpt2source = fastpm.lpt1_gradient(lpt2source, self._source['InitialPosition'], self['GradLPTDisp2'].compute())
        gradient[...] += fastpm.lpt2source_gradient(self.dlink, gradient_lpt2source)

        return gradient

    @property
    def size(self):
        return len(self._source['InitialPosition'])

    @column
    def InitialPosition(self):
        return self.make_column(self._source['InitialPosition'])

    @column
    def LPTDisp1(self):
        return self.make_column(self._source['LPTDisp1'])

    @column
    def LPTDisp2(self):
        return self.make_column(self._source['LPTDisp2'])

    @column
    def Position(self):
        return (self['InitialPosition'].astype('f4')
                + self['LPTDisp1'] * self.D1
                + self['LPTDisp2'] * self.D2)
    @column
    def Velocity(self):
        return (self['LPTDisp1'] * self.D1 * self.f1 * self.a ** 2 * self.E
                       + self['LPTDisp2'] * self.D2 * self.f2 * self.a ** 2 * self.E)
    @column
    def GradLPTDisp1(self):
        return self['GradVelocity'] * (self.D1 * self.f1 * self.a **2 * self.E) + self['GradPosition'] * self.D1

    @column
    def GradLPTDisp2(self):
        return self['GradVelocity'] * (self.D2 * self.f2 * self.a **2 * self.E) + self['GradPosition'] * self.D2
