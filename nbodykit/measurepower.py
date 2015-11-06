import numpy

from mpi4py import MPI

def measurepower(comm, k, p3d, kedges, Nmu, los='z', poles=[]):
    """ Measure power spectrum P(k,mu) from the simple 3d basis to desired 
        (k,mu) basis or one d basis.
        
        If multipole numbers are specified in the ``poles`` argument, 
        compute and return the power multipoles from P(k,mu)

        Notes
        -----
        *   the power spectrum is measured in bins of k and mu
        *   the mu range extends from 0 to 1.0
        *   the mu bins are half-inclusive half-exclusive, except the last bin
            is inclusive on both ends (to include mu = 1.0)
        *   when Nmu == 1, the case reduces to the isotropic 1D power spectrum.
        
        Parameters
        ----------
        
        comm : MPI.Comm
            the communicator for the decomposition of the power spectrum

        k  : list
            The list of k value for each item in the p3d array. The items
            must broadcast to the same shape of p3d.

        p3d : array_like (real)
            The power spectrum in 3D. 

        Nmu : int
            the number of mu bins to use when binning in the power spectrum
        
        los : str, {'x','y','z'}
            the line-of-sight direction, which the angle `mu` is defined with
            respect to. Default is `z`.
            
        poles : list of int, optional
            if provided, a list of integers specifying multipole numbers to compute
            from P(k,mu)  
    """
    from scipy.special import legendre
        
    Nk = len(kedges) - 1 
    # mu bin edges
    muedges = numpy.linspace(0, 1, Nmu+1, endpoint=True)
    
    # always measure make sure first ell is monopole, which
    # is just P(k,mu) since legendre of ell=0 is 1
    do_poles = len(poles) > 0
    poles_ = [0]+sorted(poles) if 0 not in poles else sorted(poles)
    ell_idx = [poles_.index(l) for l in poles]
    Nell = len(poles_)
    
    # valid ell values
    if any(ell < 0 for ell in poles_):
        raise RuntimeError("multipole numbers must be nonnegative integers")

    # squared k bin edges
    k2edges = kedges ** 2

    musum = numpy.zeros((Nk+2, Nmu+2))
    ksum = numpy.zeros((Nk+2, Nmu+2))
    Psum = numpy.zeros((Nell, Nk+2, Nmu+2)) # extra dimension for multipoles
    Nsum = numpy.zeros((Nk+2, Nmu+2))
    
    # los index
    los_index = 'xyz'.index(los)
    
    # count everything but z = 0 plane twice (r2c transform stores 1/2 modes)
    nonsingular = numpy.squeeze(k[2] != 0) # has length of Nz now
    
    for row in range(len(k[0])):
        
        # now scratch stores k ** 2
        scratch = numpy.float64(k[0][row] ** 2)
        for ki in k[1:]:
            scratch = scratch + ki[0] ** 2

        if len(scratch.flat) == 0:
            # no data
            continue

        dig_k = numpy.digitize(scratch.flat, k2edges)
    
        # make scratch just k
        scratch **= 0.5
    
        # store mu
        with numpy.errstate(invalid='ignore'):
            if los_index == 0:
                mu = abs(k[los_index][row]/scratch)
            else:
                mu = abs(k[los_index][0]/scratch)
        dig_mu = numpy.digitize(mu.flat, muedges)
        
        # make the multi-index
        multi_index = numpy.ravel_multi_index([dig_k, dig_mu], (Nk+2,Nmu+2))
    
        # count modes not in singular plane twice
        scratch[:, nonsingular] *= 2.
    
        # the k sum
        ksum.flat += numpy.bincount(multi_index, weights=scratch.flat, minlength=ksum.size)
    
        # take the sum of weights
        scratch[...] = 1.0
        # count modes not in singular plane twice
        scratch[:, nonsingular] = 2.
        Nsum.flat += numpy.bincount(multi_index, weights=scratch.flat, minlength=Nsum.size)

        scratch[...] = p3d[row]

        # the singular plane is down weighted by 0.5
        scratch[:, nonsingular] *= 2.
        
        # weight P(k,mu) and sum the weighted values
        for iell, ell in enumerate(poles_):
            weighted_pkmu = scratch * (2*ell + 1.) * legendre(ell)(mu)
            Psum[iell,...].flat += numpy.bincount(multi_index, weights=weighted_pkmu.flat, minlength=Nsum.size)
        
        # the mu sum
        mu[:, nonsingular] *= 2.
        musum.flat += numpy.bincount(multi_index, weights=mu.flat, minlength=musum.size)

    ksum = comm.allreduce(ksum, MPI.SUM)
    musum = comm.allreduce(musum, MPI.SUM)
    Psum = comm.allreduce(Psum, MPI.SUM)
    Nsum = comm.allreduce(Nsum, MPI.SUM)

    # add the last 'internal' mu bin (mu == 1) to the last visible mu bin
    # this makes the last visible mu bin inclusive on both ends.
    Psum[..., -2] += Psum[..., -1]
    musum[:, -2] += musum[:, -1]
    ksum[:, -2] += ksum[:, -1]
    Nsum[:, -2] += Nsum[:, -1]

    # reshape and slice to remove out of bounds points
    sl = slice(1, -1)
    with numpy.errstate(invalid='ignore'):
        
        # 2D P(k,mu) results
        pkmu = (Psum[0,...] / Nsum)[sl,sl] # ell=0 is first index
        kmean_2d = (ksum / Nsum)[sl,sl]
        mumean_2d = (musum / Nsum)[sl, sl]
        N_2d = Nsum[sl,sl]
        
        # 1D multipole results (summing over mu (last) axis)
        if do_poles:
            N_1d = Nsum[sl,sl].sum(axis=-1)
            kmean_1d = ksum[sl,sl].sum(axis=-1) / N_1d
            poles = Psum[:, sl,sl].sum(axis=-1) / N_1d
            poles = poles[ell_idx,...]
    
    # return just P(k,mu) or P(k,mu) + multipoles
    edges = [kedges, muedges]
    if not do_poles:
        return kmean_2d, mumean_2d, pkmu, N_2d, edges
    else:
        pole_result = (kmean_1d, poles, N_1d)
        pkmu_result = (kmean_2d, mumean_2d, pkmu, N_2d)
        return pole_result, pkmu_result, edges
