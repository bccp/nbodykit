import os
import numpy
import logging
import warnings

from nbodykit import CurrentMPIComm
from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource
from nbodykit import _global_options

class FFTRecon(MeshSource):
    """
    FFT based Lagrangian reconstruction algorithm in a periodic box.

    References:

        Eisenstein et al, 2007
        http://adsabs.harvard.edu/abs/2007ApJ...664..675E
        Section 3, paragraph starting with 'Restoring in full the ...'

    We follow a cleaner description in Schmitfull et al 2015,

        http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1508.06972

    Table I, and text below. Schemes are LGS, LF2 and LRR.

    A slight difference against the paper is that Redshift distortion
    and bias are corrected in the linear order. The Random shifting
    followed Martin White's suggestion to exclude the RSD by default.
    (with default `revert_rsd_random=False`.)

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
        the size of the periodic box, default is to infer from the data.

    scheme : string
        The reconstruction scheme.
        `LGS` is the standard reconstruction (Lagrangian growth shift).
        `LF2` is the F2 Lagrangian reconstruction.
        `LRR` is the random-random Lagrangian reconstruction.
    """

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
            scheme='LGS',
            BoxSize=None):

        assert scheme in ['LGS', 'LF2', 'LRR']

        assert isinstance(data, CatalogSource)
        assert isinstance(ran, CatalogSource)

        comm = data.comm

        assert data.comm == ran.comm

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

        if (self.pm.BoxSize / self.pm.Nmesh).max() > R:
            if comm.rank == 0:
                warnings.warn("The smoothing radius smaller than the mesh cell size. This may produce undesired numerical results.")

        assert position in data.columns
        assert position in ran.columns

        self.position = position

        MeshSource.__init__(self, comm, pm.Nmesh.copy(), pm.BoxSize.copy(), pm.dtype)

        self.attrs['bias'] = bias
        self.attrs['f'] = f
        self.attrs['los'] = los
        self.attrs['R'] = R
        self.attrs['scheme'] = scheme
        self.attrs['revert_rsd_random'] = bool(revert_rsd_random)

        self.data = data
        self.ran = ran

        if self.comm.rank == 0:
            self.logger.info("Reconstruction for bias=%g, f=%g, smoothing R=%g los=%s" % (self.attrs['bias'], self.attrs['f'], self.attrs['R'], str(self.attrs['los'])))
            self.logger.info("Reconstruction scheme = %s" % (self.attrs['scheme']))

    def to_real_field(self):
        return self.run()

    def run(self):

        s_d, s_r = self._compute_s()
        return self._helper_paint(s_d, s_r)

    def work_with(self, cat, s):
        pm = self.pm
        delta = pm.create(mode='real', value=0)

        # ensure the slices are synced, since decomposition is collective
        Nlocalmax = max(pm.comm.allgather(cat.size))

        # python 2.7 wants floats.
        nbar = (1.0 * cat.csize / self.pm.Nmesh.prod())

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

    def _summary_field(self, field, name):
        cmean = field.cmean()
        if self.comm.rank == 0:
            self.logger.info("painted %s, mean=%g" % (name, cmean))


    def _helper_paint(self, s_d, s_r):
        """ Convert the displacements of data and random to a single reconstruction mesh object. """

        def LGS(delta_s_r):
            delta_s_d = self.work_with(self.data, s_d)
            self._summary_field(delta_s_d, "delta_s_d (shifted)")

            delta_s_d[...] -= delta_s_r
            return delta_s_d

        def LRR(delta_s_r):
            delta_s_nr = self.work_with(self.ran, -s_r)
            self._summary_field(delta_s_nr, "delta_s_nr (reverse shifted)")

            delta_d = self.work_with(self.data, None)
            self._summary_field(delta_d, "delta_d (unshifted)")

            delta_s_nr[...] += delta_s_r[...]
            delta_s_nr[...] *= 0.5
            delta_d[...] -= delta_s_nr
            return delta_d

        def LF2(delta_s_r):
            lgs = LGS(delta_s_r)
            lrr = LRR(delta_s_r) 
            lgs[...] *= 3.0 / 7.0
            lrr[...] *= 4.0 / 7.0
            lgs[...] += lrr
            return lgs

        delta_s_r = self.work_with(self.ran, s_r)
        self._summary_field(delta_s_r, "delta_s_r (shifted)")

        if self.attrs['scheme'] == 'LGS':
            delta_recon = LGS(delta_s_r)
        elif self.attrs['scheme'] == 'LF2':
            delta_recon = LF2(delta_s_r)
        elif self.attrs['scheme'] == 'LRR':
            delta_recon = LRR(delta_s_r)
            
        self._summary_field(delta_recon, "delta_recon")

        # FIXME: perhaps change to 1 + delta for consistency. But it means loss of precision in f4 
        return delta_recon

    def _compute_s(self):
        """ Computing the reconstruction displacement of data and random """

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

        delta_d = self.work_with(self.data, None)
        self._summary_field(delta_d, "delta_d (unshifted)")
        delta_d = delta_d.r2c(out=Ellipsis)

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

