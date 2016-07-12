import numpy
from nbodykit.utils.meshtools import SlabIterator

def make_gaussian_fields(pm, linear_power, compute_displacement=False, random_state=None):
    r"""
    Make a Gaussian realization of a overdensity field, :math:`\delta(x)`
    
    If specified, also compute the corresponding linear Zel'dovich 
    displacement field :math:`\psi(x)`, which is related to the 
    linear velocity field via: 
    
    .. math::
        
        v(x) = \frac{\psi(x)}{f a H}
    
    Notes
    -----
    This computes the overdensity field using the following steps: 
    
        1. Generate random variates from :math:`\mathcal{N}(\mu=0, \sigma=1)`
        2. FFT the above field from configuration to Fourier space
        3. Scale the Fourier field by :math:`(P(k) N^3 / V)^{1/2}`
        4. FFT back to configuration space
    
    After step 2, the field has a variance of :math:`N^{-3}` (using the 
    normalization convention of `pmesh`), since the variance of the
    complex FFT (with no additional normalization) is 
    :math:`N^3 \times \sigma^2_\mathrm{real}` and `pmesh` divides each field 
    by :math:`N^3`. 
    
    Furthermore, the power spectrum is defined as V * variance. 
    Thus, the extra factor of N**3 / V that shows up in step 3, 
    cancels this factor such that the power spectrum is P(k).
    
    The linear displacement field is computed as:
    
    .. math::
    
        \psi_i(k) = i \frac{k_i}{k^2} \delta(k)
    
    .. note:: 
    
        To recover the linear velocity in proper units, i.e., km/s, 
        from the linear displacement, an additional factor of 
        :math:`f \times a \times H(a)` is required
    
    Parameters
    ----------
    pm : ParticleMesh
        the mesh instance to compute the fields on
    linear_power : callable
        a function taking wavenumber as its only argument, which returns
        the linear power spectrum
    compute_displacement : bool, optional
        if ``True``, also return the linear Zel'dovich displacement field; 
        default is ``False``
    random_state : numpy.random.RandomState, optional
        the random state used to draw normally distributed random variates
    
    Returns
    -------
    delta : array_like
        the real-space Gaussian overdensity field
    disp : array_like or ``None``
        if requested, the Gaussian displacement field
    """  
    if random_state is None: random_state = numpy.random
              
    # assign Gaussian rvs with mean 0 and unit variance
    pm.real[:] = random_state.normal(size=pm.real.shape)
    
    # initialize the displacement field arrays
    if compute_displacement:
        disp_k = numpy.repeat((numpy.ones_like(pm.complex)*1j)[None], 3, axis=0)
        disp_x = numpy.repeat(numpy.zeros_like(pm.real)[None], 3, axis=0)
    else:
        disp_x = None
        
    # FT to k-space
    pm.r2c()
    
    # normalization
    norm = pm.Nmesh**3 / pm.BoxSize.prod()
    
    # loop over the mesh, slab by slab
    for slab in SlabIterator(pm.k, axis=0):
        
        # the square of the norm of k on the mesh
        kslab = slab.norm2()
    
        # the linear power (function of k)
        power = linear_power((kslab**0.5).flatten())
            
        # multiply complex field by sqrt of power
        pm.complex[slab.index].flat *= (power*norm)**0.5
        
        # set k == 0 to zero (zero x-space mean)
        zero_idx = kslab == 0. 
        pm.complex[slab.index][zero_idx] = 0.
        
        # compute the displacement
        if compute_displacement:
            with numpy.errstate(invalid='ignore'):
                disp_k[0][slab.index] *= slab.coords(0) / kslab * pm.complex[slab.index]
                disp_k[1][slab.index] *= slab.coords(1) / kslab * pm.complex[slab.index]
                disp_k[2][slab.index] *= slab.coords(2) / kslab * pm.complex[slab.index] 
                
            # no bulk displacement
            for i in [0, 1, 2]:    
                disp_k[i][slab.index][zero_idx] = 0.       
                
    # FFT the density to real-space
    pm.c2r()
    delta = pm.real.copy()
    
    # FFT the velocity back to real space
    if compute_displacement:
        for i in [0, 1, 2]:
            pm.complex[:] = disp_k[i][:]
            pm.c2r()
            disp_x[i] = pm.real[:]
    
    # return density and displacement (which could be None)
    return delta, disp_x
    
   
def lognormal_transform(density, bias=1.):
    r"""
    Apply a (biased) lognormal transformation of the density
    field by computing:
    
    .. math::
    
        F(\delta) = \frac{1}{N} e^{b*\delta}
    
    where :math:`\delta` is the initial overdensity field and the
    normalization :math:`N` is chosen such that 
    :math:`\langle F(\delta) \rangle = 1`
    
    Parameters
    ----------
    density : array_like
        the input density field to apply the transformation to
    bias : float, optional
        optionally apply a linear bias to the density field; 
        default is unbiased (1.0)
    """
    toret = numpy.exp(bias * density)
    toret /= toret.mean()
    return toret
    
    
def poisson_sample_to_points(delta, displacement, pm, nbar, rsd=None, f=0., bias=1., random_state=None):
    """
    Poisson sample the linear delta and displacement fields to points. 
    
    This applies the following steps:
    
        1. use a (biased) lognormal transformation to make the overdensity 
           field positive-definite
        2. use the Zel'dovich displacement field to mimic nonlinear growth of structure
        3. poisson sample the overdensity field to points, disributing particles
           uniformly within the mesh cells
    
    Parameters
    ----------
    delta : array_like
        the linear overdensity field to sample
    displacement : array_like
        the linear displacement field which is used to move the particles
    pm : ParticleMesh
        the mesh instance
    nbar : float
        the desired number density of the output catalog of objects
    rsd : {'x', 'y' 'z'}
        the direction to apply RSD to; if ``None`` (default), no RSD 
        will be added
    f : float, optional
        the growth rate, equal to `f = dlnD\dlna`, which scales the 
        strength of the RSD; default is 0. (no RSD)
    bias : float, optional
        apply a linear bias to the overdensity field (default is 1.)
    random_state : numpy.random.RandomState, optional
        the random state used to perform the Poisson sampling
    
    Returns
    -------
    pos : array_like, (N, 3)
        the Cartesian positions of the particles in the box     
    """
    if random_state is None: random_state = numpy.random
    
    # apply the lognormal transformatiom to the initial conditions density
    # this creates a positive-definite delta (necessary for Poisson sampling)
    lagrangian_bias = bias - 1.
    delta = lognormal_transform(delta, bias=lagrangian_bias)
    
    # mean number of objects per cell
    H = pm.BoxSize / pm.Nmesh
    overallmean = H.prod() * nbar
    
    # number of objects in each cell
    cellmean = delta*overallmean
    
    # number of objects in each cell
    N = random_state.poisson(cellmean)
    Ntot = N.sum()
    pts = N.nonzero() # indices of nonzero points
    
    # setup the coordinate grid
    x = numpy.squeeze(pm.x[0])[pts[0]]
    y = numpy.squeeze(pm.x[1])[pts[1]]
    z = numpy.squeeze(pm.x[2])[pts[2]]
    
    # the displacement field for all nonzero grid cells
    offset = displacement[:, pts[0], pts[1], pts[2]]
    
    # add RSD to the displacement field in the specified direction
    # in Zel'dovich approx, RSD is implemented with an additional factor of (1+f)
    if rsd is not None:
        if f <= 0.:
            raise ValueError("a RSD direction was provided, but the growth rate is not positive")
        rsd_index = "xyz".index(rsd)
        offset[rsd_index] *= (1. + f)
    
    # displace the coordinate mesh
    x += offset[0]
    y += offset[1]
    z += offset[2]
    
    # coordinates of each object (placed randomly in each cell)
    x = numpy.repeat(x, N[pts]) + random_state.uniform(0, H[0], size=Ntot)
    y = numpy.repeat(y, N[pts]) + random_state.uniform(0, H[1], size=Ntot)
    z = numpy.repeat(z, N[pts]) + random_state.uniform(0, H[2], size=Ntot)
    
    # enforce periodic and stack
    x %= pm.BoxSize[0]
    y %= pm.BoxSize[1]
    z %= pm.BoxSize[2]
    pos = numpy.vstack([x, y, z]).T
    
    return pos
    
    

        
                

        
    
    
    