import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource

class FFTRecon(MeshSource):
    """
    Standard FFT based reconstruction algorithm for a periodic box.
    The algorithm does not deal with redshift distortion.

    References:

        Eisenstein et al, 2007
        http://adsabs.harvard.edu/abs/2007ApJ...664..675E
        Section 3, paragraph starting with 'Restoring in full the ...'

    However, a cleaner description is in Schmitfull et al 2015,

        http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1508.06972

        Equation 38.

    Parameters
    ----------
    data : CatalogSource,
        the data catalog, e.g. halos. `data.attrs['BoxSize']` is used if argument `BoxSize` is not given.
    ran  :  CatalogSource
        the random catalog, e.g. from a `UniformCatalog` object.
    Nmesh : int
        The size of the FFT Mesh. Rule of thumb is that the size of a mesh cell
        shall be 2 ~ 4 times smaller than the smoothing length, `R`.
    revert_rsd_random : boolean
        Revert the rsd for randoms as well as data. There are two conventions.
        either reverting rsd displacement in data displacement only(False) or
        in both data and randoms (True). Default is False.
    R : float
        The radius of smoothing. 10 to 20 Mpc/h is usually cool.
    bias : float
        The bias of the data catalog.
    f: float
        The growth rate; if non-zero, correct for RSD
    los : list 
        The direction of the line of sight for RSD. Usually (default) [0, 0, 1].
    position: string
        column to use for picking up the Position of the objects.
    BoxSize : float or array_like
        the size of the periodic box.

    """

    @CurrentMPIComm.enable
    def __init__(self,
            data,
            ran,
            Nmesh,
            bias=1.0,
            f = 0.0,
            los = [0, 0, 1],
            R=20,
            position='Position',
            revert_rsd_random=False,
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

        los = numpy.array(los, dtype='f8', copy=True)
        los /= (los ** 2).sum()

        assert len(los) == 3
        assert (~numpy.isnan(los)).all()

        pm = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh, comm=comm)
        self.pm = pm

        assert position in data.columns
        assert position in ran.columns

        self.position = position

        MeshSource.__init__(self, comm, pm.Nmesh.copy(), pm.BoxSize.copy(), pm.dtype)

        self.attrs['bias'] = bias
        self.attrs['f'] = f
        self.attrs['los'] = los
        self.attrs['R'] = R
        self.attrs['revert_rsd_random'] = bool(revert_rsd_random)

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
        """ Convert the displacements of data and random to a single reconstruction mesh object. """
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
        """ Computing the reconstruction displacement of data and random """

        nbar_d = (self.data.csize / self.pm.Nmesh.prod())
        nbar_r = (self.ran.csize / self.pm.Nmesh.prod())

        def kernel(d):
            def kernel(k, v):
                k2 = sum(ki**2 for ki in k)
                k2[k2 == 0] = 1.0

                # reverting rsd.
                mu = sum(k[i] * self.attrs['los'][i] for i in range(len(k))) / k2 ** 0.5

                v = v * numpy.exp(-0.5 * k2 * self.attrs['R']**2)

                frac = self.attrs['bias'] * ( 1 + self.attrs['f'] / self.attrs['bias'] * mu ** 2)

                v = v / frac

                return 1j * k[d] / k2 * v
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

        # convention 1: shifting data only
        s_d *= (1 + self.attrs['los'] * self.attrs['f'])

        # convention 2: shifting data only
        if self.attrs['revert_rsd_random']:
            s_r *= (1 + self.attrs['los'] * self.attrs['f'])
        
        return s_d, s_r

