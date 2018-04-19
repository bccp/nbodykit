import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource

class FFTRecon(MeshSource):
    """
    Standard FFT based reconstruction algorithm for a periodic box.
    """

    @CurrentMPIComm.enable
    def __init__(self,
            data,
            ran,
            Nmesh,
            bias=1.0,
            R=20,
            position='Position',
            BoxSize=None,
            comm=None):

        assert isinstance(data, CatalogSource)
        assert isinstance(ran, CatalogSource)

        from pmesh.pm import ParticleMesh

        if Nmesh is None:
            Nmesh = data.attrs['Nmesh']
        _Nmesh = numpy.empty(3, dtype='i8')
        _Nmesh[...] = Nmesh

        if BoxSize is None:
            BoxSize = data.attrs['BoxSize']

        pm = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh)
        self.pm = pm

        assert position in data.columns
        assert position in ran.columns

        self.position = position

        MeshSource.__init__(self, comm, pm.Nmesh.copy(), pm.BoxSize.copy(), pm.dtype)

        self.attrs['bias'] = bias
        self.attrs['R'] = R

        self.data = data
        self.ran = ran


    def to_real_field(self):
        return self.run()

    def run(self):

        dpos = self.data[self.position].compute()
        rpos = self.ran[self.position].compute()

        s_d, s_r = self._compute_s(dpos, rpos)
        return self._paint(dpos - s_d, rpos - s_r)

    def _paint(self, dpos, rpos):

        nbar_d = (self.data.csize / self.pm.Nmesh.prod())
        nbar_r = (self.ran.csize / self.pm.Nmesh.prod())

        layout = self.pm.decompose(dpos)
        rlayout = self.pm.decompose(rpos)

        delta_d = self.pm.paint(dpos, layout=layout)
        delta_d[...] /= nbar_d
        delta_r = self.pm.paint(rpos, layout=rlayout)
        delta_r[...] /= nbar_r

        return delta_d - delta_r

    def _compute_s(self, dpos, rpos):
        
        nbar_d = (self.data.csize / self.pm.Nmesh.prod())
        nbar_r = (self.ran.csize / self.pm.Nmesh.prod())

        def kernel(d):
            def kernel(k, v):
                k2 = sum(ki**2 for ki in k)
                k2[k2 == 0] = 1.0
                return 1j * k[d] / k2 * v * numpy.exp(-0.5 * k2 * self.attrs['R']**2) / self.attrs['bias']
            return kernel

        layout = self.pm.decompose(dpos)
        rlayout = self.pm.decompose(rpos)

        delta_d = self.pm.paint(dpos, layout=layout)
        delta_d[...] /= nbar_d

        delta_dc = delta_d.r2c(out=Ellipsis)

        s_d = numpy.empty_like(dpos)
        s_r = numpy.empty_like(rpos)

        for d in range(3):
            tmp = delta_dc.apply(kernel(d)).c2r(out=Ellipsis)
            s_d[..., d] = tmp.readout(dpos, layout=layout)
            s_r[..., d] = tmp.readout(rpos, layout=rlayout)

        return s_d, s_r

