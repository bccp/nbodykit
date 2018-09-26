import numpy
import logging
import time
import warnings

from nbodykit import CurrentMPIComm
from nbodykit.utils import timer
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.algorithms.fftpower import project_to_basis, _find_unique_edges
from pmesh.pm import ComplexField

def get_real_Ylm(l, m):
    """
    Return a function that computes the real spherical
    harmonic of order (l,m)

    Parameters
    ----------
    l : int
        the degree of the harmonic
    m : int
        the order of the harmonic; abs(m) <= l

    Returns
    -------
    Ylm : callable
        a function that takes 4 arguments: (xhat, yhat, zhat)
        unit-normalized Cartesian coordinates and returns the
        specified Ylm

    References
    ----------
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    """
    import sympy as sp

    # make sure l,m are integers
    l = int(l); m = int(m)

    # the relevant cartesian and spherical symbols
    x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
    xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
    phi, theta = sp.symbols('phi theta')
    defs = [(sp.sin(phi), y/sp.sqrt(x**2+y**2)),
            (sp.cos(phi), x/sp.sqrt(x**2+y**2)),
            (sp.cos(theta), z/sp.sqrt(x**2 + y**2+z**2))]

    # the normalization factors
    if m == 0:
        amp = sp.sqrt((2*l+1) / (4*numpy.pi))
    else:
        amp = sp.sqrt(2*(2*l+1) / (4*numpy.pi) * sp.factorial(l-abs(m)) / sp.factorial(l+abs(m)))

    # the cos(theta) dependence encoded by the associated Legendre poly
    expr = (-1)**m * sp.assoc_legendre(l, abs(m), sp.cos(theta))

    # the phi dependence
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m)*phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m*phi))

    # simplify
    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x/r, xhat), (y/r, yhat), (z/r, zhat)])
    Ylm = sp.lambdify((xhat,yhat,zhat), expr, 'numexpr')

    # attach some meta-data
    Ylm.expr = expr
    Ylm.l    = l
    Ylm.m    = m

    return Ylm

class ConvolvedFFTPower(object):
    """
    Algorithm to compute power spectrum multipoles using FFTs
    for a data survey with non-trivial geometry.

    Due to the geometry, the estimator computes the true power spectrum
    convolved with the window function (FFT of the geometry).

    This estimator implemented in this class is described in detail in
    Hand et al. 2017 (arxiv:1704.02357). It uses the spherical harmonic
    addition theorem such that only :math:`2\ell+1` FFTs are required to
    compute each multipole. This differs from the implementation in
    Bianchi et al. and Scoccimarro et al., which requires
    :math:`(\ell+1)(\ell+2)/2` FFTs.

    Results are computed when the object is inititalized, and the result is
    stored in the :attr:`poles` attribute. Important meta-data computed
    during algorithm execution is stored in the :attr:`attrs` dict. See the
    documenation of :func:`~ConvolvedFFTPower.run`.

    .. note::
        A full tutorial on the class is available in the documentation
        :ref:`here <convpower>`.

    .. note::
        Cross correlations are only supported when the FKP weight column
        differs between the two mesh objects, i.e., the underlying ``data``
        and ``randoms`` must be the same. This allows users to compute
        the cross power spectrum of the same density field, weighted
        differently.

    Parameters
    ----------
    first : FKPCatalog, FKPCatalogMesh
        the first source to paint the data/randoms; FKPCatalog is automatically
        converted to a FKPCatalogMesh, using default painting parameters
    poles : list of int
        a list of integer multipole numbers ``ell`` to compute
    second : FKPCatalog, FKPCatalogMesh, optional
        the second source to paint the data/randoms; cross correlations are
        only supported when the weight column differs between the two mesh
        objects, i.e., the underlying ``data`` and ``randoms`` must be the same!
    kmin : float, optional
        the edge of the first wavenumber bin; default is 0
    dk : float, optional
        the spacing in wavenumber to use; if not provided; the fundamental mode
        of the box is used

    References
    ----------
    * Hand, Nick et al. `An optimal FFT-based anisotropic power spectrum estimator`, 2017
    * Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
      MNRAS, 2015
    * Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
    """
    logger = logging.getLogger('ConvolvedFFTPower')

    def __init__(self, first, poles,
                    second=None,
                    Nmesh=None,
                    kmin=0.,
                    dk=None,
                    use_fkp_weights=None,
                    P0_FKP=None):

        if use_fkp_weights is not None or P0_FKP is not None:
            raise ValueError("use_fkp_weights and P0_FKP are deprecated. Assign a FKPWeight column to source['randoms']['FKPWeight'] and source['data']['FKPWeight'] with the help of the FKPWeightFromNbar(nbar) function")

        first = _cast_mesh(first, Nmesh=Nmesh)
        if second is not None:
            second = _cast_mesh(second, Nmesh=Nmesh)
        else:
            second = first

        # data/randoms of second must be same as second
        # only difference can be FKP weight currently
        if not is_valid_crosscorr(first, second):
            msg = ("ConvolvedFFTPower cross-correlations currently require the same"
                   " FKPCatalog (data/randoms), such that only the weight column can vary")
            raise NotImplementedError(msg)

        self.first = first
        self.second = second

        # grab comm from first source
        self.comm = first.comm

        # check for comm mismatch
        assert second.comm is first.comm, "communicator mismatch between input sources"

        # make a box big enough for both catalogs if they are not equal
        # NOTE: both first/second must have the same BoxCenter to recenter Position
        if not numpy.array_equal(first.attrs['BoxSize'], second.attrs['BoxSize']):

            # stack box coordinates together
            joint = {}
            for name in ['BoxSize', 'BoxCenter']:
                joint[name] = numpy.vstack([first.attrs[name], second.attrs[name]])

            # determine max box length along each dimension
            argmax = numpy.argmax(joint['BoxSize'], axis=0)
            joint['BoxSize'] = joint['BoxSize'][argmax, [0,1,2]]
            joint['BoxCenter'] = joint['BoxCenter'][argmax, [0,1,2]]

            # re-center the box
            first.recenter_box(joint['BoxSize'], joint['BoxCenter'])
            second.recenter_box(joint['BoxSize'], joint['BoxCenter'])

        # make a list of multipole numbers
        if numpy.isscalar(poles):
            poles = [poles]

        # store meta-data
        self.attrs = {}
        self.attrs['poles'] = poles
        self.attrs['dk'] = dk
        self.attrs['kmin'] = kmin

        # store BoxSize and BoxCenter from source
        self.attrs['Nmesh'] = self.first.attrs['Nmesh'].copy()
        self.attrs['BoxSize'] = self.first.attrs['BoxSize']
        self.attrs['BoxPad'] = self.first.attrs['BoxPad']
        self.attrs['BoxCenter'] = self.first.attrs['BoxCenter']

        # grab some mesh attrs, too
        self.attrs['mesh.resampler'] = self.first.resampler
        self.attrs['mesh.interlaced'] = self.first.interlaced

        # and run
        self.run()

    def run(self):
        """
        Compute the power spectrum multipoles. This function does not return
        anything, but adds several attributes (see below).

        Attributes
        ----------
        edges : array_like
            the edges of the wavenumber bins
        poles : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object that behaves similar to a structured array, with
            fancy slicing and re-indexing; it holds the measured multipole
            results, as well as the number of modes (``modes``) and average
            wavenumbers values in each bin (``k``)
        attrs : dict
            dictionary holding input parameters and several important quantites
            computed during execution:

            #. data.N, randoms.N :
                the unweighted number of data and randoms objects
            #. data.W, randoms.W :
                the weighted number of data and randoms objects, using the
                column specified as the completeness weights
            #. alpha :
                the ratio of ``data.W`` to ``randoms.W``
            #. data.norm, randoms.norm :
                the normalization of the power spectrum, computed from either
                the "data" or "randoms" catalog (they should be similar).
                See equations 13 and 14 of arxiv:1312.4611.
            #. data.shotnoise, randoms.shotnoise :
                the shot noise values for the "data" and "random" catalogs;
                See equation 15 of arxiv:1312.4611.
            #. shotnoise :
                the total shot noise for the power spectrum, equal to
                ``data.shotnoise`` + ``randoms.shotnoise``; this should be subtracted from
                the monopole.
            #. BoxSize :
                the size of the Cartesian box used to grid the data and
                randoms objects on a Cartesian mesh.

            For further details on the meta-data, see
            :ref:`the documentation <fkp-meta-data>`.
        """
        pm = self.first.pm

        # setup the binning in k out to the minimum nyquist frequency
        dk = 2*numpy.pi/pm.BoxSize.min() if self.attrs['dk'] is None else self.attrs['dk']

        if dk > 0:
            kedges = numpy.arange(self.attrs['kmin'], numpy.pi*pm.Nmesh.min()/pm.BoxSize.max() + dk/2, dk)
            kcoords = None
        else:
            k = pm.create_coords('complex')
            kedges, kcoords = _find_unique_edges(k, 2 * numpy.pi / pm.BoxSize, pm.comm)
            if self.comm.rank == 0:
                self.logger.info('%d unique k values are found' % len(kcoords))

        # measure the binned 1D multipoles in Fourier space
        result = self._compute_multipoles(kedges)

        # set all the necessary results
        self.poles = BinnedStatistic(['k'], [kedges], result,
                            fields_to_sum=['modes'],
                            coords=[kcoords],
                            **self.attrs)
        self.edges = kedges

    def to_pkmu(self, mu_edges, max_ell):
        """
        Invert the measured multipoles :math:`P_\ell(k)` into power
        spectrum wedges, :math:`P(k,\mu)`.

        Parameters
        ----------
        mu_edges : array_like
            the edges of the :math:`\mu` bins
        max_ell : int
            the maximum multipole to use when computing the wedges;
            all even multipoles with :math:`ell` less than or equal
            to this number are included

        Returns
        -------
        pkmu : BinnedStatistic
            a data set holding the :math:`P(k,\mu)` wedges
        """
        from scipy.special import legendre
        from scipy.integrate import quad

        def compute_coefficient(ell, mumin, mumax):
            """
            Compute how much each multipole contributes to a given wedges.
            This returns:

            .. math::
                \frac{1}{\mu_{max} - \mu_{max}} \int_{\mu_{min}}^{\mu^{max}} \mathcal{L}_\ell(\mu)
            """
            norm = 1.0 / (mumax - mumin)
            return norm * quad(lambda mu: legendre(ell)(mu), mumin, mumax)[0]

        # make sure we have all the poles measured
        ells = list(range(0, max_ell+1, 2))
        if any('power_%d' %ell not in self.poles for ell in ells):
            raise ValueError("measurements for ells=%s required if max_ell=%d" %(ells, max_ell))

        # new data array
        dtype = numpy.dtype([('power', 'c8'), ('k', 'f8'), ('mu', 'f8')])
        data = numpy.zeros((self.poles.shape[0], len(mu_edges)-1), dtype=dtype)

        # loop over each wedge
        bounds = list(zip(mu_edges[:-1], mu_edges[1:]))
        for imu, mulims in enumerate(bounds):

            # add the contribution from each Pell
            for ell in ells:
                coeff = compute_coefficient(ell, *mulims)
                data['power'][:,imu] += coeff * self.poles['power_%d' %ell]

            data['k'][:,imu] = self.poles['k']
            data['mu'][:,imu] = numpy.ones(len(data))*0.5*(mulims[1]+mulims[0])

        dims = ['k', 'mu']
        edges = [self.poles.edges['k'], mu_edges]
        return BinnedStatistic(dims=dims, edges=edges, data=data, coords=[self.poles.coords['k'], None], **self.attrs)

    def __getstate__(self):
        state = dict(poles=self.poles.__getstate__(),
                     attrs=self.attrs)
        return state

    def __setstate__(self, state):
        self.attrs = state['attrs']
        self.poles = BinnedStatistic.from_state(state['poles'])

    def __setstate_pre000305__(self, state):
        """ compatible version of setstate for files generated before 0.3.5 """
        edges = state['edges']
        sefl.attrs = state['attrs']
        self.poles = BinnedStatistic(['k'], [edges], self.poles, fields_to_sum=['modes'])

    def save(self, output):
        """
        Save the ConvolvedFFTPower result to disk.

        The format is currently json.

        Parameters
        ----------
        output : str
            the name of the file to dump the JSON results to
        """
        import json
        from nbodykit.utils import JSONEncoder

        # only the master rank writes
        if self.comm.rank == 0:
            self.logger.info('saving ConvolvedFFTPower result to %s' %output)

            with open(output, 'w') as ff:
                json.dump(self.__getstate__(), ff, cls=JSONEncoder)

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, output, comm=None, format='current'):
        """
        Load a saved ConvolvedFFTPower result, which has been saved to
        disk with :func:`ConvolvedFFTPower.save`.

        The current MPI communicator is automatically used
        if the ``comm`` keyword is ``None``

        format can be 'current', or 'pre000305' for files generated before 0.3.5.

        """
        import json
        from nbodykit.utils import JSONDecoder

        if comm.rank == 0:
            with open(output, 'r') as ff:
                state = json.load(ff, cls=JSONDecoder)
        else:
            state = None
        state = comm.bcast(state)
        self = object.__new__(cls)

        if format == 'current':
            self.__setstate__(state)
        elif format == 'pre000305':
            self.__setstate_pre000305__(state)

        self.comm = comm
        return self

    def _compute_multipoles(self, kedges):
        """
        Compute the window-convoled power spectrum multipoles, for a data set
        with non-trivial survey geometry.

        This estimator builds upon the work presented in Bianchi et al. 2015
        and Scoccimarro et al. 2015, but differs in the implementation. This
        class uses the spherical harmonic addition theorem such that
        only :math:`2\ell+1` FFTs are required per multipole, rather than the
        :math:`(\ell+1)(\ell+2)/2` FFTs in the implementation presented by
        Bianchi et al. and Scoccimarro et al.

        References
        ----------
        * Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
          MNRAS, 2015
        * Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
        """
        # clear compensation from the actions
        for source in [self.first, self.second]:
            source.actions[:] = []; source.compensated = False
            assert len(source.actions) == 0

        # compute the compensations
        compensation = {}
        for name, mesh in zip(['first', 'second'], [self.first, self.second]):
            compensation[name] = get_compensation(mesh)
            if self.comm.rank == 0:
                if compensation[name] is not None:
                    args = (compensation[name]['func'].__name__, name)
                    self.logger.info("using compensation function %s for source '%s'" % args)
                else:
                    self.logger.warning("no compensation applied for source '%s'" % name)

        rank = self.comm.rank
        pm   = self.first.pm

        # setup the 1D-binning
        muedges = numpy.linspace(0, 1, 2, endpoint=True)
        edges = [kedges, muedges]

        # make a structured array to hold the results
        cols   = ['k'] + ['power_%d' %l for l in sorted(self.attrs['poles'])] + ['modes']
        dtype  = ['f8'] + ['c8']*len(self.attrs['poles']) + ['i8']
        dtype  = numpy.dtype(list(zip(cols, dtype)))
        result = numpy.empty(len(kedges)-1, dtype=dtype)

        # offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to
        # the original (x,y,z) coords
        offset = self.attrs['BoxCenter'] + 0.5*pm.BoxSize / pm.Nmesh

        # always need to compute ell=0
        poles = sorted(self.attrs['poles'])
        if 0 not in poles:
            poles = [0] + poles
        assert poles[0] == 0

        # spherical harmonic kernels (for ell > 0)
        Ylms = [[get_real_Ylm(l,m) for m in range(-l, l+1)] for l in poles[1:]]

        # paint the 1st FKP density field to the mesh (paints: data - alpha*randoms, essentially)
        rfield1 = self.first.compute(Nmesh=self.attrs['Nmesh'])
        meta1 = rfield1.attrs.copy()
        if rank == 0:
            self.logger.info("%s painting of 'first' done" %self.first.resampler)

        # store alpha: ratio of data to randoms
        self.attrs['alpha'] = meta1['alpha']

        # FFT 1st density field and apply the resampler transfer kernel
        cfield = rfield1.r2c()
        if compensation['first'] is not None:
            cfield.apply(out=Ellipsis, **compensation['first'])
        if rank == 0: self.logger.info('ell = 0 done; 1 r2c completed')

        # monopole A0 is just the FFT of the FKP density field
        # NOTE: this holds FFT of density field #1
        volume = pm.BoxSize.prod()
        A0_1 = ComplexField(pm)
        A0_1[:] = cfield[:] * volume # normalize with a factor of volume

        # paint second mesh too?
        if self.first is not self.second:

            # paint the second field
            rfield2 = self.second.compute(Nmesh=self.attrs['Nmesh'])
            meta2 = rfield2.attrs.copy()
            if rank == 0: self.logger.info("%s painting of 'second' done" %self.second.resampler)

            # need monopole of second field
            if 0 in self.attrs['poles']:

                # FFT density field and apply the resampler transfer kernel
                A0_2 = rfield2.r2c()
                A0_2[:] *= volume
                if compensation['second'] is not None:
                    A0_2.apply(out=Ellipsis, **compensation['second'])
        else:
            rfield2 = rfield1
            meta2 = meta1

            # monopole of second field is first field
            if 0 in self.attrs['poles']:
                A0_2 = A0_1

        # ensure alpha from first mesh is equal to alpha from second mesh
        # NOTE: this is mostly just a sanity check, and should always be true if
        # we made it this far already
        if not numpy.allclose(rfield1.attrs['alpha'], rfield2.attrs['alpha'], rtol=1e-3):
            msg = ("ConvolvedFFTPower cross-correlations currently require the same"
                   " FKPCatalog (data/randoms), such that only the weight column can vary;"
                   " different ``alpha`` values found for first/second meshes")
            raise ValueError(msg)

        # save the painted density field #2 for later
        density2 = rfield2.copy()

        # initialize the memory holding the Aell terms for
        # higher multipoles (this holds sum of m for fixed ell)
        # NOTE: this will hold FFTs of density field #2
        Aell = ComplexField(pm)

        # the real-space grid
        xgrid = [xx.astype('f8') + offset[ii] for ii, xx in enumerate(density2.slabs.optx)]
        xnorm = numpy.sqrt(sum(xx**2 for xx in xgrid))
        xgrid = [x/xnorm for x in xgrid]

        # the Fourier-space grid
        kgrid = [kk.astype('f8') for kk in cfield.slabs.optx]
        knorm = numpy.sqrt(sum(kk**2 for kk in kgrid)); knorm[knorm==0.] = numpy.inf
        kgrid = [k/knorm for k in kgrid]

        # proper normalization: same as equation 49 of Scoccimarro et al. 2015
        for name in ['data', 'randoms']:
            self.attrs[name+'.norm'] = self.normalization(name, self.attrs['alpha'])

        if self.attrs['randoms.norm'] > 0:
            norm = 1.0 / self.attrs['randoms.norm']

            # check normalization
            Adata = self.attrs['data.norm']
            Aran = self.attrs['randoms.norm']
            if not numpy.allclose(Adata, Aran, rtol=0.05):
                msg = "normalization in ConvolvedFFTPower different by more than 5%; "
                msg += ",algorithm requires they must be similar\n"
                msg += "\trandoms.norm = %.6f, data.norm = %.6f\n" % (Aran, Adata)
                msg += "\tpossible discrepancies could be related to normalization "
                msg += "of n(z) column ('%s')\n" % self.first.nbar
                msg += "\tor the consistency of the FKP weight column for 'data' "
                msg += "and 'randoms';\n"
                msg += "\tn(z) columns for 'data' and 'randoms' should be "
                msg += "normalized to represent n(z) of the data catalog"
                raise ValueError(msg)

            if rank == 0:
                self.logger.info("normalized power spectrum with `randoms.norm = %.6f`" % Aran)
        else:
            # an empty random catalog is provides, so we will ignore the normalization.
            norm = 1.0
            if rank == 0:
                self.logger.info("normalization of power spectrum is neglected, as no random is provided.")


        # loop over the higher order multipoles (ell > 0)
        start = time.time()
        for iell, ell in enumerate(poles[1:]):

            # clear 2D workspace
            Aell[:] = 0.

            # iterate from m=-l to m=l and apply Ylm
            substart = time.time()
            for Ylm in Ylms[iell]:

                # reset the real-space mesh to the original density #2
                rfield2[:] = density2[:]

                # apply the config-space Ylm
                for islab, slab in enumerate(rfield2.slabs):
                    slab[:] *= Ylm(xgrid[0][islab], xgrid[1][islab], xgrid[2][islab])

                # real to complex of field #2
                rfield2.r2c(out=cfield)

                # apply the Fourier-space Ylm
                for islab, slab in enumerate(cfield.slabs):
                    slab[:] *= Ylm(kgrid[0][islab], kgrid[1][islab], kgrid[2][islab])

                # add to the total sum
                Aell[:] += cfield[:]

                # and this contribution to the total sum
                substop = time.time()
                if rank == 0:
                    self.logger.debug("done term for Y(l=%d, m=%d) in %s" %(Ylm.l, Ylm.m, timer(substart, substop)))

            # apply the compensation transfer function
            if compensation['second'] is not None:
                Aell.apply(out=Ellipsis, **compensation['second'])

            # factor of 4*pi from spherical harmonic addition theorem + volume factor
            Aell[:] *= 4*numpy.pi*volume

            # log the total number of FFTs computed for each ell
            if rank == 0:
                args = (ell, len(Ylms[iell]))
                self.logger.info('ell = %d done; %s r2c completed' %args)

            # calculate the power spectrum multipoles, slab-by-slab to save memory
            # NOTE: this computes (A0 of field #1) * (Aell of field #2).conj()
            for islab in range(A0_1.shape[0]):
                Aell[islab,...] = norm * A0_1[islab] * Aell[islab].conj()

            # project on to 1d k-basis (averaging over mu=[0,1])
            proj_result, _ = project_to_basis(Aell, edges)
            result['power_%d' %ell][:] = numpy.squeeze(proj_result[2])

        # summarize how long it took
        stop = time.time()
        if rank == 0:
            self.logger.info("higher order multipoles computed in elapsed time %s" %timer(start, stop))

        # also compute ell=0
        if 0 in self.attrs['poles']:

            # the 3D monopole
            for islab in range(A0_1.shape[0]):
                A0_1[islab,...] = norm*A0_1[islab]*A0_2[islab].conj()

            # the 1D monopole
            proj_result, _ = project_to_basis(A0_1, edges)
            result['power_0'][:] = numpy.squeeze(proj_result[2])

        # save the number of modes and k
        result['k'][:] = numpy.squeeze(proj_result[0])
        result['modes'][:] = numpy.squeeze(proj_result[-1])

        # compute shot noise
        self.attrs['shotnoise'] = self.shotnoise(self.attrs['alpha'])

        # copy over any painting meta data
        if self.first is self.second:
            copy_meta(self.attrs, meta1)
        else:
            copy_meta(self.attrs, meta1, prefix='first')
            copy_meta(self.attrs, meta2, prefix='second')

        return result

    def normalization(self, name, alpha):
        r"""
        Compute the power spectrum normalization, using either the
        ``data`` or ``randoms`` source.

        The normalization is given by:

        .. math::

            A = \int d^3x \bar{n}'_1(x) \bar{n}'_2(x) w_{\mathrm{fkp},1} w_{\mathrm{fkp},2}.

        The mean densities are assumed to be the same, so this can be converted
        to a summation over objects in the source, as

        .. math::

            A = \sum w_{\mathrm{comp},1} \bar{n}_2 w_{\mathrm{fkp},1} w_{\mathrm{fkp},2}.

        References
        ----------
        see Eqs. 13,14 of Beutler et al. 2014, "The clustering of galaxies in the
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift
        space distortions using the power spectrum multipoles"
        """
        assert name in ['data', 'randoms']

        if name+'.norm' not in self.attrs:

            # the selection (same for first/second)
            sel = self.first.source.compute(self.first.source[name][self.first.selection])

            # selected first/second meshes for "name" (data or randoms)
            first = self.first.source[name][sel]
            second = self.second.source[name][sel]

            # these are assumed the same b/w first and second meshes
            comp_weight = first[self.first.comp_weight]
            nbar = second[self.second.nbar]

            # different weights allowed for first and second mesh
            fkp_weight1 = first[self.first.fkp_weight]
            if self.second is self.first:
                fkp_weight2 = fkp_weight1
            else:
                fkp_weight2 = second[self.second.fkp_weight]

            A  = nbar*comp_weight*fkp_weight1*fkp_weight2
            if name == 'randoms':
                A *= alpha
            A = first.compute(A.sum())
            self.attrs[name+'.norm'] = self.comm.allreduce(A)

        return self.attrs[name+'.norm']

    def shotnoise(self, alpha):
        r"""
        Compute the power spectrum shot noise, using either the
        ``data`` or ``randoms`` source.

        This computes:

        .. math::

            S = \sum (w_\mathrm{comp} w_\mathrm{fkp})^2

        References
        ----------
        see Eq. 15 of Beutler et al. 2014, "The clustering of galaxies in the
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift
        space distortions using the power spectrum multipoles"
        """
        if 'shotnoise' not in self.attrs:

            Pshot = 0
            for name in ['data', 'randoms']:

                # the selection (same for first/second)
                sel = self.first.source.compute(self.first.source[name][self.first.selection])

                # selected first/second meshes for "name" (data or randoms)
                first = self.first.source[name][sel]
                second = self.second.source[name][sel]

                # completeness weights (assumed same for first/second)
                comp_weight = first[self.first.comp_weight]

                # different weights allowed for first and second mesh
                fkp_weight1 = first[self.first.fkp_weight]
                if self.first is self.second:
                    fkp_weight2 = fkp_weight1
                else:
                    fkp_weight2 = second[self.second.fkp_weight]

                S = (comp_weight**2*fkp_weight1*fkp_weight2).sum()
                if name == 'randoms':
                    S *= alpha**2
                Pshot += S # add to total

        # reduce sum across all ranks
        Pshot = self.comm.allreduce(first.compute(Pshot))

        # divide by normalization from randoms
        return Pshot / self.attrs['randoms.norm']

def _cast_mesh(mesh, Nmesh):
    """
    Cast an object to a MeshSource. Nmesh is used only on FKPCatalog

    """
    from .catalog import FKPCatalog
    from .catalogmesh import FKPCatalogMesh

    if not isinstance(mesh, (FKPCatalogMesh, FKPCatalog)):
        raise TypeError("input sources should be a FKPCatalog or FKPCatalogMesh")

    if isinstance(mesh, FKPCatalog):
        # if input is CatalogSource, use defaults to make it into a mesh
        mesh = mesh.to_mesh(Nmesh=Nmesh, dtype='f8', compensated=False)

    if Nmesh is not None and any(mesh.attrs['Nmesh'] != Nmesh):
        raise ValueError(("Mismatched Nmesh between __init__ and mesh.attrs; "
                          "if trying to re-sample with a different mesh, specify "
                          "`Nmesh` as keyword of to_mesh()"))

    return mesh

def get_compensation(mesh):
    toret = None
    try:
        compensation = mesh._get_compensation()
        toret = {'func':compensation[0][1], 'kind':compensation[0][2]}
    except ValueError:
        pass
    return toret

def copy_meta(attrs, meta, prefix=""):
    if prefix:
        prefix += '.'
    for key in meta:
        if key.startswith('data.') or key.startswith('randoms.'):
            attrs[prefix+key] = meta[key]

def is_valid_crosscorr(first, second):

    if second.source is not first.source:
        return False

    same_cols = ['selection', 'comp_weight', 'nbar']
    if any(getattr(second, name) != getattr(first, name) for name in same_cols):
        return False

    return True
