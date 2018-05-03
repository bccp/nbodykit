import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource
from nbodykit import _global_options

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

        if self.comm.rank == 0:
            self.logger.info("Reconstruction for bias=%g, f=%g, smoothing R=%g los=%s\n" % (self.attrs['bias'], self.attrs['f'], self.attrs['R'], str(self.attrs['los'])))

    def to_real_field(self):
        return self.run()

    def run(self):

        s_d, s_r = self._compute_s()
        return self._paint(s_d, s_r)

    def work_with(self, cat, s):
        pm = self.pm
        delta = pm.create(mode='real', value=0)

        # ensure the slices are synced, since decomposition is collective
        Nlocalmax = max(pm.comm.allgather(cat.size))

        nbar = (cat.csize / self.pm.Nmesh.prod())

        chunksize = _global_options['paint_chunk_size']

        for i in range(0, Nlocalmax, chunksize):
            sl = slice(i, i + chunksize)

            if s is not None:
                dpos = (cat[self.position].astype('f4')[sl] - s[sl]).compute()
            else:
                dpos = (cat[self.position].astype('f4')[sl]).compute()

            layout = self.pm.decompose(dpos)
            self.pm.paint(dpos, layout=layout, out=delta, hold=True)

        delta[...] /= nbar

        return delta

    def _paint(self, s_d, s_r):
        """ Convert the displacements of data and random to a single reconstruction mesh object. """

        delta_d = self.work_with(self.data, s_d)
        delta_d_mean = delta_d.cmean()
        if self.comm.rank == 0:
            self.logger.info("painted delta_d, mean=%g" % delta_d_mean)

        delta_r = self.work_with(self.ran, s_r)
        delta_r_mean = delta_r.cmean()
        if self.comm.rank == 0:
            self.logger.info("painted delta_r, mean=%g" % delta_r_mean)

        delta_d[...] -= delta_r
        recon_mean = delta_d.cmean()
        if self.comm.rank == 0:
            self.logger.info("painted reconstructed field, mean=%g" % recon_mean)
        # FIXME: perhaps change to 1 + delta for consistency. But it means loss of precision in f4 
        return delta_d

    def _compute_s(self):
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

        delta_d = self.work_with(self.data, None).r2c(out=Ellipsis)

        def solve_displacement(cat, delta_d):
            dpos = cat[self.position].astype('f4').compute()
            layout = self.pm.decompose(dpos)
            s_d = numpy.zeros_like(dpos, dtype='f4')

            for d in range(3):
                delta_d.apply(kernel(d)).c2r(out=Ellipsis) \
                       .readout(dpos, layout=layout, out=s_d[..., d])
            return s_d

        s_d = solve_displacement(self.data, delta_d)
        s_d_std = (self.comm.allreduce((s_d ** 2).sum(axis=0)) / self.data.csize) ** 0.5
        if self.comm.rank == 0:
            self.logger.info("Solved displacements of data, std(s_d) = %s" % str(s_d_std))

        s_r = solve_displacement(self.ran, delta_d)
        s_r_std = (self.comm.allreduce((s_r ** 2).sum(axis=0)) / self.ran.csize) ** 0.5
        if self.comm.rank == 0:
            self.logger.info("Solved displacements of randoms, std(s_r) = %s" % str(s_r_std))

        # convention 1: shifting data only
        s_d[...] *= (1 + self.attrs['los'] * self.attrs['f'])

        # convention 2: shifting data only
        if self.attrs['revert_rsd_random']:
            s_r[...] *= (1 + self.attrs['los'] * self.attrs['f'])
        
        return s_d, s_r

