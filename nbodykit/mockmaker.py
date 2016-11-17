import numpy
from pmesh.pm import RealField, ComplexField
from nbodykit.utils.meshtools import SlabIterator

def gaussian_complex_fields(pm, linear_power, compute_displacement=False):
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
    pm : pmesh.pm.ParticleMesh
        the mesh object
    linear_power : callable
        a function taking wavenumber as its only argument, which returns
        the linear power spectrum
    compute_displacement : bool, optional
        if ``True``, also return the linear Zel'dovich displacement field; 
        default is ``False``
    
    Returns
    -------
    delta_k : ComplexField
        the real-space Gaussian overdensity field
    disp_k : ComplexField or ``None``
        if requested, the Gaussian displacement field
    """            
    # assign Gaussian rvs with mean 0 and unit variance
    delta = RealField(pm)
    delta[:] = numpy.random.normal(size=delta.shape)
            
    # FT to k-space
    delta_k = delta.r2c()
    
    # initialize the displacement fields for (x,y,z)
    if compute_displacement:
        disp_k = [ComplexField(pm) for i in [0,1,2]]
        for i in [0,1,2]: disp_k[i][:] = 1j
    else:
        disp_k = None
    
    # normalization
    norm = delta.Nmesh.prod() / delta.BoxSize.prod()
    
    # loop over the mesh, slab by slab
    for slab in SlabIterator(pm.k, axis=0):
        
        # the square of the norm of k on the mesh
        kslab = slab.norm2()
    
        # the linear power (function of k)
        power = linear_power((kslab**0.5).flatten())
            
        # multiply complex field by sqrt of power
        delta_k[slab.index].flat *= (power*norm)**0.5
        
        # set k == 0 to zero (zero x-space mean)
        zero_idx = kslab == 0. 
        delta_k[slab.index][zero_idx] = 0.
        
        # compute the displacement
        if compute_displacement:
            with numpy.errstate(invalid='ignore'):
                this_delta = delta_k[slab.index]
                disp_k[0][slab.index] *= slab.coords(0) / kslab * this_delta
                disp_k[1][slab.index] *= slab.coords(1) / kslab * this_delta
                disp_k[2][slab.index] *= slab.coords(2) / kslab * this_delta
                
            # no bulk displacement
            for i in [0, 1, 2]:    
                disp_k[i][slab.index][zero_idx] = 0.       
                    
    # return Fourier-space density and displacement (which could be None)
    return delta_k, disp_k
    
    
def gaussian_real_fields(pm, linear_power, compute_displacement=False):
    r"""
    Make a Gaussian realization of a overdensity field in 
    real-space :math:`\delta(x)`
    
    If specified, also compute the corresponding linear Zel'dovich 
    displacement field :math:`\psi(x)`, which is related to the 
    linear velocity field via: 
    
    Notes
    -----
    See the docstring for :func:`gaussian_complex_fields` for the
    steps involved in generating the fields
    
    Parameters
    ----------
    pm : pmesh.pm.ParticleMesh
        the mesh object
    linear_power : callable
        a function taking wavenumber as its only argument, which returns
        the linear power spectrum
    compute_displacement : bool, optional
        if ``True``, also return the linear Zel'dovich displacement field; 
        default is ``False``
    
    Returns
    -------
    delta : RealField
        the real-space Gaussian overdensity field
    disp : RealField or ``None``
        if requested, the Gaussian displacement field
    """
    # make fourier fields
    delta_k, disp_k = gaussian_complex_fields(pm, linear_power, compute_displacement)
                
    # FFT the density to real-space
    delta = delta_k.c2r()
    
    # FFT the velocity back to real space
    if compute_displacement:
        disp = [disp_k[i].c2r() for i in [0, 1, 2]]
    else:
        disp = None

    # return density and displacement (which could be None)
    return delta, disp
    
   
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
    
    Returns
    -------
    toret : RealField
        the real field holding the transformed density field
    """
    toret = density.copy()
    toret[:] = numpy.exp(bias * density.value)
    toret[:] /= numpy.mean(toret)
    return toret
    
    
def poisson_sample_to_points(delta, displacement, pm, nbar, rsd=None, f=0., bias=1.):
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
    delta : RealField
        the linear overdensity field to sample
    displacement : list of RealField (3,)
        the linear displacement fields which is used to move the particles
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
    
    Returns
    -------
    pos : array_like, (N, 3)
        the Cartesian positions of the particles in the box     
    """
    # apply the lognormal transformation to the initial conditions density
    # this creates a positive-definite delta (necessary for Poisson sampling)
    lagrangian_bias = bias - 1.
    delta = lognormal_transform(delta, bias=lagrangian_bias)
    
    # mean number of objects per cell
    H = delta.BoxSize / delta.Nmesh
    overallmean = H.prod() * nbar
    
    # number of objects in each cell
    cellmean = delta*overallmean
    
    # number of objects in each cell
    N = numpy.random.poisson(cellmean)
    Ntot = N.sum()
    pts = N.nonzero() # indices of nonzero points
    
    # setup the coordinate grid
    x = numpy.squeeze(delta.pm.x[0])[pts[0]]
    y = numpy.squeeze(delta.pm.x[1])[pts[1]]
    z = numpy.squeeze(delta.pm.x[2])[pts[2]]
    
    # the displacement field for all nonzero grid cells
    offset = [displacement[i][pts[0], pts[1], pts[2]] for i in [0,1,2]]
    
    # add RSD to the displacement field in the specified direction
    # in Zel'dovich approx, RSD is implemented with an additional factor of (1+f)
    if rsd is not None:
        if f <= 0.:
            raise ValueError("a RSD direction was provided, but the growth rate is not positive")
        if rsd not in ["x", "y", "z"]:
            raise ValueError("'rsd' should be one of ['x', 'y', 'z']")
        rsd_index = "xyz".index(rsd)
        offset[rsd_index] *= (1. + f)
    
    # displace the coordinate mesh
    x += offset[0]
    y += offset[1]
    z += offset[2]
    
    # coordinates of each object (placed randomly in each cell)
    x = numpy.repeat(x, N[pts]) + numpy.random.uniform(0, H[0], size=Ntot)
    y = numpy.repeat(y, N[pts]) + numpy.random.uniform(0, H[1], size=Ntot)
    z = numpy.repeat(z, N[pts]) + numpy.random.uniform(0, H[2], size=Ntot)
    
    # enforce periodic and stack
    x %= delta.BoxSize[0]
    y %= delta.BoxSize[1]
    z %= delta.BoxSize[2]
    return numpy.vstack([x, y, z]).T
  