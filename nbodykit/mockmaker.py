import numpy
import numbers
from mpi4py import MPI

from pmesh.pm import RealField, ComplexField
from nbodykit.meshtools import SlabIterator
from nbodykit.utils import GatherArray, ScatterArray

def gaussian_complex_fields(pm, linear_power, seed, remove_variance=False, compute_displacement=False):
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
    
        1. Generate complex variates with unity variance
        2. Scale the Fourier field by :math:`(P(k) / V)^{1/2}`

    After step 2, the complex field has unity variance. This 
    is equivalent to generating real-space normal variates
    with mean and unity variance, calling r2c() and dividing by :math:`N^3`
    since the variance of the complex FFT (with no additional normalization) 
    is :math:`N^3 \times \sigma^2_\mathrm{real}`.
    
    Furthermore, the power spectrum is defined as V * variance. 
    So a normalization factor of 1 / V shows up in step 2, 
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
    seed : int
        the random seed used to generate the random field
    compute_displacement : bool, optional
        if ``True``, also return the linear Zel'dovich displacement field; 
        default is ``False``
    remove_variance : bool, optional
        if ``True``, remove the variance in the amplitude of the guassian,
        such that the power spectrum of the realization is exactly the input.

    Returns
    -------
    delta_k : ComplexField
        the real-space Gaussian overdensity field
    disp_k : ComplexField or ``None``
        if requested, the Gaussian displacement field
    """    
    if not isinstance(seed, numbers.Integral):
        raise ValueError("the seed used to generate the linear field must be an integer")        
        
    # use pmesh to generate random complex white noise field (done in parallel)
    # variance of complex field is unity
    # multiply by P(k)**0.5 to get desired variance
    delta_k = pm.generate_whitenoise(seed, mode='complex', unitary=remove_variance)
        
    # initialize the displacement fields for (x,y,z)
    if compute_displacement:
        disp_k = [ComplexField(pm) for i in range(delta_k.ndim)]
        for i in range(delta_k.ndim): disp_k[i][:] = 1j
    else:
        disp_k = None
    
    # volume factor needed for normalization
    norm = 1.0 / pm.BoxSize.prod()
    
    # iterate in slabs over fields
    slabs = [delta_k.slabs.x, delta_k.slabs]
    if compute_displacement: 
        slabs += [d.slabs for d in disp_k]
    
    # loop over the mesh, slab by slab
    for islabs in zip(*slabs):
        kslab, delta_slab = islabs[:2] # the k arrays and delta slab
        
        # the square of the norm of k on the mesh
        k2 = sum(kk**2 for kk in kslab)
    
        # the linear power (function of k)
        power = linear_power((k2**0.5).flatten())
            
        # multiply complex field by sqrt of power
        delta_slab[...].flat *= (power*norm)**0.5
        
        # set k == 0 to zero (zero config-space mean)
        zero_idx = k2 == 0. 
        delta_slab[zero_idx] = 0.
        
        # compute the displacement
        if compute_displacement:
            
            # ignore division where k==0 and set to 0
            with numpy.errstate(invalid='ignore'):
                for i in range(delta_k.ndim):                    
                    disp_slab = islabs[2+i]
                    disp_slab[...] *= kslab[i] / k2 * delta_slab[...]
                    disp_slab[zero_idx] = 0. # no bulk displacement
                    
    # return Fourier-space density and displacement (which could be None)
    return delta_k, disp_k
    
    
def gaussian_real_fields(pm, linear_power, seed, compute_displacement=False):
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
    seed : int
        the random seed used to generate the random field
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
    delta_k, disp_k = gaussian_complex_fields(pm, linear_power, seed, compute_displacement=compute_displacement)
                
    # FFT the density to real-space
    delta = delta_k.c2r()
    
    # FFT the velocity back to real space
    if compute_displacement:
        disp = [disp_k[i].c2r() for i in range(delta.ndim)]
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
    
    
def poisson_sample_to_points(delta, displacement, pm, nbar, f=0., bias=1., seed=None, comm=None):
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
    f : float, optional
        the growth rate, equal to `f = dlnD\dlna`, which scales the 
        strength of the RSD; default is 0. (no RSD)
    bias : float, optional
        apply a linear bias to the overdensity field (default is 1.)
    seed : int, optional
        the random seed used to Poisson sample the field to points
    
    Returns
    -------
    pos : array_like, (N, 3)
        the Cartesian positions of each of the generated particles
    vel : array_like, (N, 3)
        the velocity "offsets" of each of the generated particles in the same
        units as the ``pos`` array
    """
    if comm is None:
        comm = MPI.COMM_WORLD
        
    # create a random state with the input seed
    rng = numpy.random.RandomState(seed)
    
    # apply the lognormal transformation to the initial conditions density
    # this creates a positive-definite delta (necessary for Poisson sampling)
    lagrangian_bias = bias - 1.
    delta = lognormal_transform(delta, bias=lagrangian_bias)
    
    # mean number of objects per cell
    H = delta.BoxSize / delta.Nmesh
    overallmean = H.prod() * nbar
    
    # number of objects in each cell (per rank)
    cellmean = delta.value*overallmean
    cellmean = GatherArray(cellmean, comm, root=0)
    
    # rank 0 computes the poisson sampling
    if comm.rank == 0:
        N = rng.poisson(cellmean)
    else:
        N = None
    
    # scatter N back evenly across the ranks
    counts = comm.allgather(delta.shape[0])
    N = ScatterArray(N, comm, root=0, counts=counts)
    
    Nlocal = N.sum() # local number of particles
    Ntot = comm.allreduce(Nlocal) # the collective number of particles
    nonzero_cells = N.nonzero() # indices of nonzero cells

    # initialize the mesh of particle positions and velocity
    # this has the shape: (number of dimensions, number of nonzero cells)
    pos_mesh = numpy.empty(numpy.shape(nonzero_cells), dtype=delta.dtype)
    vel_mesh = numpy.empty_like(pos_mesh)
    
    # generate the coordinates for each nonzero cell
    for i in range(delta.ndim): 
        
        # particle positions initially on the coordinate grid
        pos_mesh[i] = numpy.squeeze(delta.pm.x[i])[nonzero_cells[i]]
        
        # velocity offsets for each particle
        vel_mesh[i] = displacement[i][nonzero_cells]
        
        # displace the positions by the velocity offset
        pos_mesh[i] += vel_mesh[i]
        
    # convert displacement field to RSD
    # in Zel'dovich approx, RSD is implemented with an additional factor of (1+f)
    if f <= 0.:
        raise ValueError("a RSD direction was provided, but the growth rate is not positive")
    
    # rank 0 computes the in-cell uniform offsets
    if comm.rank == 0:
        in_cell_shift = numpy.empty((Ntot, delta.ndim), dtype=delta.dtype)
        for i in range(delta.ndim):
            in_cell_shift[:,i] = rng.uniform(0, H[i], size=Ntot)
    else:
        in_cell_shift = None
    
    # scatter the in-cell uniform offsets back to the ranks
    counts = comm.allgather(Nlocal) 
    in_cell_shift = ScatterArray(in_cell_shift, comm, root=0, counts=counts)
    
    # initialize the output array of particle positions and velocity
    # this has shape: (local number of particles, number of dimensions)
    pos = numpy.zeros((Nlocal, delta.ndim), dtype=delta.dtype)
    vel = numpy.zeros_like(pos)
    
    # coordinates of each object (placed randomly in each cell)
    for i in range(delta.ndim):
        pos[:,i] = numpy.repeat(pos_mesh[i], N[nonzero_cells]) + in_cell_shift[:,i]
        pos[:,i] %= delta.BoxSize[i]

    # velocities of each object
    for i in range(delta.ndim):
        vel_mesh[i] *= (1. + f) # scale to redshift with extra factor of f
        vel[:,i] = numpy.repeat(vel_mesh[i], N[nonzero_cells])

    return pos, vel
  
