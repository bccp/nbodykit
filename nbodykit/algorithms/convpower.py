import numpy
import logging
import time
import warnings

from nbodykit import CurrentMPIComm
from nbodykit.utils import timer
from nbodykit.dataset import DataSet

from .fftpower import project_to_basis
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
        the order of the harmonic; |m| < l
    
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
    Algorithm to compute the power spectrum multipoles using FFTs
    for a data survey with non-trivial geometry. 
    
    Due to the geometry, the estimator computes the true power spectrum
    convolved with the window function (FFT of the geometry).
    
    This estimator builds upon the work presented in Bianchi et al. 2015
    and Scoccimarro et al. 2015, but differs in the implementation. This
    class uses the spherical harmonic addition theorem such that
    only :math:`2\ell+1` FFTs are required per multipole, rather than the
    :math:`(\ell+1)(\ell+2)/2` FFTs in the implementation presented by
    Bianchi et al. and Scoccimarro et al.
    
    Thanks to Yin Li for pointing out the spherical harmonic decomposition.
    
    References
    ----------
    * Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
      MNRAS, 2015
    * Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
    """
    logger = logging.getLogger('ConvolvedFFTPower')

    def __init__(self, source, poles, 
                    Nmesh=None, 
                    kmin=0., 
                    dk=None, 
                    use_fkp_weights=False, 
                    P0_FKP=None):
        """
        Parameters
        ----------
        source : FKPCatalog, FKPMeshSource
            the source to paint the data/randoms; FKPCatalog is automatically converted
            to a FKPMeshSource, using default painting parameters
        poles : list of int
            a list of integer multipole numbers ``ell`` to compute
        kmin : float; optional
            the edge of the first wavenumber bin; default is 0
        dk : float; optional
            the spacing in wavenumber to use; if not provided; the fundamental mode of the
            box is used
        use_fkp_weights : bool; optional
            if ``True``, FKP weights will be added using ``P0_FKP`` such that the 
            fkp weight is given by ``1 / (1 + P0*NZ)`` where ``NZ`` is the number density
            as a function of redshift column
        P0_FKP : float; optional
            the value of ``P0`` to use when computing FKP weights
        """
        if not hasattr(source, 'paint'):
            source = source.to_mesh(Nmesh=Nmesh)
        self.source  = source
        self.Nmesh   = Nmesh
        self.comm = self.source.comm
        
        # make a list of multipole numbers
        if numpy.isscalar(poles):
            poles = [poles]
        
        if use_fkp_weights and P0_FKP is None:
            raise ValueError(("please set the 'P0_FKP' keyword if you wish to automatically "
                              "use FKP weights with 'use_fkp_weights=True'"))
                
        # add FKP weights
        if use_fkp_weights:
            if self.comm.rank == 0:
                self.logger.info("adding FKP weights as the '%s' column, using P0 = %.4e" %(self.source.fkp_weight, P0_FKP))
            
            for name in ['data', 'randoms']:
                
                # print a warning if we are overwriting a non-default column
                old_fkp_weights = self.source[name+'.'+self.source.fkp_weight]
                if self.source.compute(old_fkp_weights.sum()) != len(old_fkp_weights):
                    warn = "it appears that we are overwriting FKP weights for the '%s' " %name
                    warn += "source in FKPCatalog when using 'use_fkp_weights=True' in ConvolvedFFTPower"
                    warnings.warn(warn)
                
                nbar = self.source[name+'.'+self.source.nbar]
                self.source[name+'.'+self.source.fkp_weight] = 1.0 / (1. + P0_FKP * nbar)
                
        self.attrs = {}
        self.attrs['poles']           = poles
        self.attrs['dk']              = dk
        self.attrs['kmin']            = kmin
        self.attrs['use_fkp_weights'] = use_fkp_weights
        self.attrs['P0_FKP']          = P0_FKP

        # store BoxSize and BoxCenter from source
        self.attrs['BoxSize']   = self.source.attrs['BoxSize']
        self.attrs['BoxPad']    = self.source.attrs['BoxPad']
        self.attrs['BoxCenter'] = self.source.attrs['BoxCenter']
        
        # grab some mesh attrs, too
        self.attrs['mesh.window']     = self.source.attrs['window']
        self.attrs['mesh.interlaced'] = self.source.attrs['interlaced']
        
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
        poles : :class:`~nbodykit.dataset.DataSet`
            a DataSet object that behaves similar to a structured array, with
            fancy slicing and re-indexing; it holds the measured multipole
            results, as well as the number of modes (``modes``) and average
            wavenumbers values in each bin (``k``)
        """
        pm = self.source.pm
        
        # setup the binning in k out to the minimum nyquist frequency
        dk = 2*numpy.pi/pm.BoxSize.min() if self.attrs['dk'] is None else self.attrs['dk']
        self.edges = numpy.arange(self.attrs['kmin'], numpy.pi*pm.Nmesh.min()/pm.BoxSize.max() + dk/2, dk)

        # measure the binned 1D multipoles in Fourier space
        poles = self._compute_multipoles()
        
        # set all the necessary results
        self.poles = DataSet(['k'], [self.edges], poles, fields_to_sum=['modes'])
    
    def to_pkmu(self, mu_edges, max_ell):
        """
        Invert the measured multipoles :math:`P_\ell(k)` into power
        spectrum wedges, :math:`P(k,\mu)`
        
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
        pkmu : DataSet
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
        return DataSet(dims=dims, edges=edges, data=data, **self.attrs)
            
    def __getstate__(self):
        state = dict(edges=self.edges,
                     poles=self.poles.data,
                     attrs=self.attrs)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.poles = DataSet(['k'], [self.edges], self.poles, fields_to_sum=['modes'])

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
    def load(cls, output, comm=None):
        """
        Load a saved ConvolvedFFTPower result, which has been saved to 
        disk with :func:`ConvolvedFFTPower.save`
        
        The current MPI communicator is automatically used
        if the ``comm`` keyword is ``None``
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
        self.__setstate__(state)
        self.comm = comm
        return self
        
    def _compute_multipoles(self):
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
        rank = self.comm.rank
        pm   = self.source.pm
        
        # setup the 1D-binning
        muedges = numpy.linspace(0, 1, 2, endpoint=True)
        edges = [self.edges, muedges]
        
        # make a structured array to hold the results
        cols   = ['k'] + ['power_%d' %l for l in sorted(self.attrs['poles'])] + ['modes']
        dtype  = ['f8'] + ['c8']*len(self.attrs['poles']) + ['i8']
        dtype  = numpy.dtype(list(zip(cols, dtype)))
        result = numpy.empty(len(self.edges)-1, dtype=dtype)
        
        # offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to 
        # the original (x,y,z) coords
        offset = self.attrs['BoxCenter'] + 0.5*pm.BoxSize / pm.Nmesh
        
        # always need to compute ell=0
        poles = sorted(self.attrs['poles'])
        if 0 not in poles:
            poles = [0] + poles
        assert poles[0] == 0
                
        # initialize the compensation transfer
        compensation = None
        try:
            compensation = self.source._get_compensation()
            if self.comm.rank == 0:
                self.logger.info('using compensation function %s' %compensation[0][1].__name__)
        except ValueError as e:
            if self.comm.rank == 0:
                self.logger.warning('no compensation applied: %s' %str(e))
            
        # spherical harmonic kernels (for ell > 0)
        Ylms = [[get_real_Ylm(l,m) for m in range(-l, l+1)] for l in poles[1:]]
                
        # paint the FKP density field to the mesh (paints: data - alpha*randoms, essentially)
        rfield = self.source.to_real_field() # just paint the real field (without any additional compensation)
        meta = rfield.attrs.copy()
        if rank == 0: self.logger.info('%s painting done' %self.source.window)
        
        # first, check if normalizations from data and randoms are similar
        # if not, n(z) column is probably wrong 
        if not numpy.allclose(meta['data.A'], meta['randoms.A'], rtol=0.05):
            msg = "normalization in ConvolvedFFTPower different by more than 5%; algorithm requires they must be similar\n"
            msg += "\trandoms.A = %.6f, data.A = %.6f\n" %(meta['randoms.A'], meta['data.A'])
            msg += "\tpossible discrepancies could be related to normalization of n(z) column ('%s')\n" %self.source.nbar
            msg += "\tor the consistency of the FKP weight column ('%s') for 'data' and 'randoms';\n" %self.source.fkp_weight
            msg += "\tn(z) columns for 'data' and 'randoms' should be normalized to represent n(z) of the data catalog"
            raise ValueError(msg)

        # save the painted density field for later
        density = rfield.copy()
        
        # FFT density field and apply the paintbrush window transfer kernel
        cfield = rfield.r2c()
        if compensation is not None:
            cfield.apply(func=compensation[0][1], kind=compensation[0][2], out=Ellipsis)
        if rank == 0: self.logger.info('ell = 0 done; 1 r2c completed')
        
        # monopole A0 is just the FFT of the FKP density field
        volume = pm.BoxSize.prod()
        A0 = ComplexField(pm)
        A0[:] = cfield[:] * volume # normalize with a factor of volume
            
        # initialize the memory holding the Aell terms for
        # higher multipoles (this holds sum of m for fixed ell)
        Aell = ComplexField(pm)
        
        # the real-space grid
        xgrid = [xx.astype('f8') + offset[ii] for ii, xx in enumerate(density.slabs.optx)]
        xnorm = numpy.sqrt(sum(xx**2 for xx in xgrid))
        xgrid = [x/xnorm for x in xgrid]
        
        # the Fourier-space grid
        kgrid = [kk.astype('f8') for kk in cfield.slabs.optx]
        knorm = numpy.sqrt(sum(kk**2 for kk in kgrid)); knorm[knorm==0.] = numpy.inf
        kgrid = [k/knorm for k in kgrid]
        
        # proper normalization: same as equation 49 of Scoccimarro et al. 2015 
        norm = 1. / meta['randoms.A']    
        
        # loop over the higher order multipoles (ell > 0)
        start = time.time()
        for iell, ell in enumerate(poles[1:]):
            
            # clear 2D workspace
            Aell[:] = 0.
                 
            # iterate from m=-l to m=l and apply Ylm
            substart = time.time()
            for Ylm in Ylms[iell]:
                
                # reset the real-space mesh to the original density
                rfield[:] = density[:]        
                
                # apply the config-space Ylm
                for islab, slab in enumerate(rfield.slabs):
                    slab[:] *= Ylm(xgrid[0][islab], xgrid[1][islab], xgrid[2][islab])
                    
                # real to complex
                rfield.r2c(out=cfield)

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
            if compensation is not None:
                Aell.apply(func=compensation[0][1], kind=compensation[0][2], out=Ellipsis)
            
            # factor of 4*pi from spherical harmonic addition theorem + volume factor
            Aell[:] *= 4*numpy.pi*volume
            
            # log the total number of FFTs computed for each ell
            if rank == 0: 
                args = (ell, len(Ylms[iell]))
                self.logger.info('ell = %d done; %s r2c completed' %args)
        
            # calculate the power spectrum multipoles, slab-by-slab to save memory 
            # this computes Aell.conj() * A0
            for islab in range(A0.shape[0]):
                Aell[islab,...] = norm*Aell[islab].conj()*A0[islab]
                
            # project on to 1d k-basis (averaging over mu=[0,1])
            proj_result, _ = project_to_basis(Aell, edges)
            result['power_%d' %ell][:] = numpy.squeeze(proj_result[2])
                
        # summarize how long it took
        stop = time.time()
        if rank == 0:
            self.logger.info("higher order multipoles computed in elapsed time %s" %timer(start, stop))
        
        # save the number of modes and k
        result['k'][:] = numpy.squeeze(proj_result[0])
        result['modes'][:] = numpy.squeeze(proj_result[-1])
        
        # also compute ell=0
        if 0 in self.attrs['poles']:
            
            # the 3D monopole
            for islab in range(A0.shape[0]):
                A0[islab,...] = norm*A0[islab]*A0[islab].conj()
            
            # the 1D monopole
            proj_result, _ = project_to_basis(A0, edges)
            result['power_0'][:] = numpy.squeeze(proj_result[2])
                
        # update with the attributes computed while painting
        self.attrs['alpha'] = meta['alpha']
        self.attrs['shotnoise'] = meta['shotnoise']
        for key in meta:
            if key.startswith('data.') or key.startswith('randoms.'):
                self.attrs[key] = meta[key]
    
        if rank == 0:
            self.logger.info("normalized power spectrum with randoms.A = %.6f" %meta['randoms.A'])

        return result
