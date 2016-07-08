import numpy

def make_gaussian_fields(pm, isotropic_power, random, compute_velocity=True):
    """
    Make a Gaussian realization of a density field, and if desired, the
    corresponding linear velocity fields
    """
    # clear the mesh
    pm.clear()
            
    # assign Gaussian rvs with mean 0 and unit variance
    pm.real[:] = random.normal(size=pm.real.shape)
    
    # initialize the velocity arrays
    if compute_velocity:
        vel_k = numpy.repeat((numpy.ones_like(pm.complex)*1j)[None], 3, axis=0)
        vel_x = numpy.repeat(numpy.zeros_like(pm.real)[None], 3, axis=0)
        
    # FT to k-space
    pm.r2c()
    
    # normalization
    norm = pm.Nmesh**3 / pm.BoxSize.prod()
    
    # loop over yz planes
    for islab in range(len(pm.k[0])):
        
        # now kslab stores k**2
        kslab = numpy.float64(pm.k[0][islab] ** 2)
        for ki in pm.k[1:]:
            kslab = kslab + ki[0]**2

        if len(kslab.flat) == 0:
            continue
    
        # the isotropic power (function of k)
        power = isotropic_power((kslab**0.5).flatten())
            
        # multiply complex field by sqrt of power
        pm.complex[islab,...].flat *= (power*norm)**0.5
        
        # set k == 0 to zero (zero x-space mean)
        zero_idx = kslab == 0. 
        pm.complex[islab, zero_idx] = 0.
        
        # do the velocity
        if compute_velocity:
            with numpy.errstate(invalid='ignore'):
                vel_k[0, islab, ...] *= pm.k[0][islab] / kslab * pm.complex[islab]
                vel_k[1, islab, ...] *= pm.k[1][0] / kslab * pm.complex[islab]
                vel_k[2, islab, ...] *= pm.k[2][0] / kslab * pm.complex[islab] 
                
            # no bulk velocity
            for i in range(3):    
                vel_k[i, islab, zero_idx] = 0.       
        
    # FT to x-space
    pm.c2r()
    delta = pm.real.copy()
    
    if not compute_velocity:
        return delta
    else:
        for i in range(3):
            pm.complex[:] = vel_k[i][:]
            pm.c2r()
            vel_x[i] = pm.real[:]
    
        return delta, vel_x
    

    
def lognormal_transform(density):
    """
    Apply a lognormal transformation to the density
    """
    toret = numpy.exp(density)
    toret /= toret.mean()
    return toret
    
    

def poisson_sample_mesh(pm, nbar, density, velocity=None):
    """
    Poisson sample the mesh
    """
    # mean number of objects per cell
    overallmean = pm.BoxSize.prod() / pm.Nmesh**3 * nbar
    
    # number of objects in each cell
    cellmean = density*overallmean
    
    # number of objects in each cell
    N = numpy.random.poisson(cellmean)
    Ntot = N.sum()
    pts = N.nonzero() # indices of nonzero points
    
    # setup the coordinate grid
    x = numpy.squeeze(pm.x[0])[pts[0]]
    y = numpy.squeeze(pm.x[1])[pts[1]]
    z = numpy.squeeze(pm.x[2])[pts[2]]
    
    # coordinates of each object (placed randomly in each cell)
    H = pm.BoxSize / pm.Nmesh
    x = numpy.repeat(x, N[pts]) + numpy.random.uniform(0, H[0], size=Ntot)
    y = numpy.repeat(y, N[pts]) + numpy.random.uniform(0, H[1], size=Ntot)
    z = numpy.repeat(z, N[pts]) + numpy.random.uniform(0, H[2], size=Ntot)
    
    # enforce periodic 
    x %= pm.BoxSize[0]
    y %= pm.BoxSize[1]
    z %= pm.BoxSize[2]
    pos = numpy.vstack([x, y, z]).T
    
    if velocity is not None:
        vx = numpy.repeat(velocity[0][pts], N[pts])
        vy = numpy.repeat(velocity[1][pts], N[pts])
        vz = numpy.repeat(velocity[2][pts], N[pts])
        vel = numpy.vstack([vx, vy, vz]).T
        
        return pos, vel
    else:
        return pos
    
    

        
                

        
    
    
    