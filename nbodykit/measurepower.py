import numpy

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction
from mpi4py import MPI

def measurepower(pm, c1, c2, Nmu, binshift=0.0, shotnoise=0.0, los='z', dk=None, kmin=0, poles=[]):
    """ Measure power spectrum P(k,mu) from density field painted on pm, and if 
        multipole numbers are specified in the ``poles`` argument, compute and
        return the power multipoles from P(k,mu)

        Notes
        -----
        *   the power spectrum is measured in bins of k and mu
        *   if no k binning is specified via the `dk` and `kmin` keywords, the bins 
            extend from 0 to the smallest 1D Nyquist (Nmesh / 2), with the units 
            consistent with the units of the BoxSize
        *   the mu range extends from 0 to 1.0
        *   the mu bins are half-inclusive half-exclusive, except the last bin
            is inclusive on both ends (to include mu = 1.0)
        *   when Nmu == 1, the case reduces to the isotropic 1D power spectrum.
        
        Parameters
        ----------
        pm : ParticleMesh
            A particle mesh object

        c1: array_like
            the complex fourier space field to measure power from.

        c2: array_like
            the complex fourier space field to measure power from.

        Nmu : int
            the number of mu bins to use when binning in the power spectrum
        
        binshift   : float
            shift the center of bins by this fraction if a bin width

        shotnoise : 0.0
            remove the shot noise contribution from the final power spectrum
            The shot noise is (pm.BoxSize) ** 3 / Ntot, where Ntot is the total
            number of particles.
        
        los : str, {'x','y','z'}
            the line-of-sight direction, which the angle `mu` is defined with
            respect to. Default is `z`.
            
        dk : float, optional
            use this spacing for k bins; if not provided, the fundamental mode
            of the box :math: `2 pi / BoxSize` is used
        
        kmin : float, optional
            the edge of the first k bin to use; default is 0
            
        poles : list of int, optional
            if provided, a list of integers specifying multipole numbers to compute
            from P(k,mu)  
    """
    from scipy.special import legendre
    
    # kedges out to the minimum nyquist frequency (accounting for possibly anisotropic box)
    BoxSize_min = numpy.amin(pm.BoxSize)
    w_to_k = pm.Nmesh / BoxSize_min
    if dk is None: 
        dk = 2*numpy.pi/BoxSize_min
    kedges = numpy.arange(kmin, numpy.pi*w_to_k + dk/2, dk)
    kedges += binshift * dk
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
    nonsingular = numpy.squeeze(pm.k[2] != 0) # has length of Nz now
    
    for row in range(len(pm.k[0])):
        
        # now scratch stores k ** 2
        scratch = numpy.float64(pm.k[0][row] ** 2)
        for ki in pm.k[1:]:
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
                mu = abs(pm.k[los_index][row]/scratch)
            else:
                mu = abs(pm.k[los_index][0]/scratch)
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

        # the power P(k,mu)
        scratch[...] = c1[row].real * c2[row].real + c1[row].imag * c2[row].imag
        # the singular plane is down weighted by 0.5
        scratch[:, nonsingular] *= 2.
        
        # weight P(k,mu) and sum the weighted values
        for iell, ell in enumerate(poles_):
            weighted_pkmu = scratch * (2*ell + 1.) * legendre(ell)(mu)
            Psum[iell,...].flat += numpy.bincount(multi_index, weights=weighted_pkmu.flat, minlength=Nsum.size)
        
        # the mu sum
        mu[:, nonsingular] *= 2.
        musum.flat += numpy.bincount(multi_index, weights=mu.flat, minlength=musum.size)

    ksum = pm.comm.allreduce(ksum, MPI.SUM)
    musum = pm.comm.allreduce(musum, MPI.SUM)
    Psum = pm.comm.allreduce(Psum, MPI.SUM)
    Nsum = pm.comm.allreduce(Nsum, MPI.SUM)

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
                   
    # each complex field has units of L^3, so power is L^6
    pkmu *= pm.BoxSize.prod() 
    if do_poles: poles *= pm.BoxSize.prod()
    pkmu -= shotnoise
    
    # return just P(k,mu) or P(k,mu) + multipoles
    edges = [kedges, muedges]
    if not do_poles:
        return kmean_2d, mumean_2d, pkmu, N_2d, edges
    else:
        pole_result = (kmean_1d, poles, N_1d)
        pkmu_result = (kmean_2d, mumean_2d, pkmu, N_2d)
        return pole_result, pkmu_result, edges
