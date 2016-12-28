from __future__ import print_function

import numpy
import logging
from nbodykit.base.particles import ParticleSource

from mpi4py import MPI
import warnings

from nbodykit import fastpm

def removeradiation(cosmo):
    if cosmo.Ogamma0 != 0:
        warnings.warn("Cosmology has radiation. Radiation is removed from background for a consistent integration.")

    return cosmo.clone(Tcmb0=0)

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
        self.dlink = dlink

        self._source = {}
        self._source['InitialPosition'] = fastpm.create_grid(self.basepm, shift=0.0)
        self._source['LPTDisp1'] = fastpm.lpt1(dlink, self._source['InitialPosition'])

        ParticleSource.__init__(self, comm=comm)

        import dask.array as da
        self._fallbacks['GradPosition'] = da.zeros((self.size, 3), dtype='f4', chunks=100000)
        self._fallbacks['GradVelocity'] = da.zeros((self.size, 3), dtype='f4', chunks=100000)
        self.set_redshift(redshift)

    def set_redshift(self, redshift):
        self.redshift = redshift

    @staticmethod
    def gradient(dlink, self):
        """ Backtrace the gradient of LPT source at dlink, using
            self['GradLPTDisp1'] as the gradient column of LPT1 displacement
        """
        return fastpm.lpt1_gradient(self.dlink, self._source['InitialPosition'], self['GradLPTDisp1'].compute())

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

