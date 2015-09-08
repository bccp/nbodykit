import numpy

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction
from mpi4py import MPI

def measurepower(pm, c1, c2, Nmu, binshift=0.0, shotnoise=0.0, los='z', dk=None, kmin=0):
    """ Measure power spectrum P(k,mu) from density field painted on pm 

        The power spectrum is measured in bins of k and mu. The k bins extend 
        from 0 to the smallest 1D Nyquist (Nmesh / 2), with the units consistent with 
        the units of the BoxSize. The mu range extends from 0 to 1.0. 
        The mu bins are half-inclusive half-exclusive, except the last bin
        is inclusive on both ends (to include mu = 1.0).

        Notes
        -----
        when Nmu == 1, the case reduces to the isotropic 1D power spectrum.
        
        Parameters
        ----------
        pm : ParticleMesh
            A particle mesh object

        c1: array_like
            the complex fourier space field to measure power from.

        c2: array_like
            the complex fourier space field to measure power from.

        Nmu : int
            The number of mu bins to use when binning in the power spectrum, 
            default is 5
        
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
            
    """
    # kedges out to the minimum nyquist frequency (accounting for possibly anisotropic box)
    BoxSize_min = numpy.amin(pm.BoxSize)
    w_to_k = pm.Nmesh / BoxSize_min
    if dk is None: 
        dk = 2*numpy.pi/BoxSize_min
    kedges = numpy.arange(kmin, numpy.pi*w_to_k + dk/2, dk)
    kedges += binshift * dk
        
    # mu bin edges
    muedges = numpy.linspace(0, 1, Nmu+1, endpoint=True)
    
    # store for convenience
    Nfreq = len(kedges) - 1 
    ndims = (Nfreq+2, Nmu+2)
    
    # freq bin edges
    k2edges = kedges ** 2

    musum = numpy.zeros(ndims)
    ksum = numpy.zeros(ndims)
    Psum = numpy.zeros(ndims)
    Nsum = numpy.zeros(ndims)
    
    # los index
    los_index = 'xyz'.index(los)
    
    for row in range(len(pm.k[0])):
        # pickup the singular plane that is single counted (r2c transform)
        singular = pm.k[2][-1] == 0

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
        multi_index = numpy.ravel_multi_index([dig_k, dig_mu], ndims)
    
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5
        mu[singular] *= 0.5
    
        # the k sum
        ksum.flat += numpy.bincount(multi_index, weights=scratch.flat, minlength=ksum.size)
    
        # the mu sum
        musum.flat += numpy.bincount(multi_index, weights=mu.flat, minlength=musum.size)
    
        # take the sum of weights
        scratch[...] = 1.0
        # the singular plane is down weighted by 0.5
        scratch[singular] = 0.5

        Nsum.flat += numpy.bincount(multi_index, weights=scratch.flat, minlength=Nsum.size)

        # take the sum of power
        scratch[...] = c1[row].real * c2[row].real + c1[row].imag * c2[row].imag
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5
        Psum.flat += numpy.bincount(multi_index, weights=scratch.flat, minlength=Psum.size)

    ksum = pm.comm.allreduce(ksum, MPI.SUM)
    musum = pm.comm.allreduce(musum, MPI.SUM)
    Psum = pm.comm.allreduce(Psum, MPI.SUM)
    Nsum = pm.comm.allreduce(Nsum, MPI.SUM)

    # add the last 'internal' mu bin (mu == 1) to the last visible mu bin
    # this makes the last visible mu bin inclusive on both ends.

    Psum[:, -2] += Psum[:, -1]
    musum[:, -2] += musum[:, -1]
    ksum[:, -2] += ksum[:, -1]
    Nsum[:, -2] += Nsum[:, -1]

    # reshape and slice to remove out of bounds points
    with numpy.errstate(invalid='ignore'):
        power = (Psum / Nsum)[1:-1, 1:-1]
        kmean = (ksum / Nsum)[1:-1, 1:-1]
        mumean = (musum / Nsum)[1:-1, 1:-1]
        N = 2*Nsum[1:-1, 1:-1] # factor of 2 for modes with negative z not in r2c transform

    # each complex field has units of L^3, so power is L^6
    power *= pm.BoxSize.prod() 
    power -= shotnoise
    return kmean, mumean, power, N, [kedges, muedges]


def measurepoles(pm, c1, c2, ells, binshift=0.0, los='z', dk=None, kmin=0):
    """ Measure power spectrum multipoles from a density field painted on pm 

        The multipoles are measured in bins of k for each multipole number ``ell`` 
        specified in ``ells``. Unless specific k bins are specified, the bins 
        extend from 0 to the smallest 1D Nyquist (Nmesh / 2), with the units consistent 
        with the units of the BoxSize. 

        
        Parameters
        ----------
        pm : ParticleMesh
            A particle mesh object

        c1: array_like
            the complex fourier space field to measure power from.

        c2: array_like
            the complex fourier space field to measure power from.

        ells : list of int
            the multipole numbers to compute
        
        binshift   : float
            shift the center of bins by this fraction if a bin width
        
        los : str, {'x','y','z'}
            the line-of-sight direction, which the angle `mu` is defined with
            respect to. Default is `z`.
            
        dk : float, optional
            use this spacing for k bins; if not provided, the fundamental mode
            of the box :math: `2 pi / BoxSize` is used
        
        kmin : float, optional
            the edge of the first k bin to use; default is 0 
    """
    from scipy.special import legendre
    
    # kedges out to the minimum nyquist frequency (accounting for possibly anisotropic box)
    BoxSize_min = numpy.amin(pm.BoxSize)
    w_to_k = pm.Nmesh / BoxSize_min
    if dk is None: 
        dk = 2*numpy.pi/BoxSize_min
    kedges = numpy.arange(kmin, numpy.pi*w_to_k + dk/2, dk)
    kedges += binshift * dk
            
    # store for convenience
    Nk = len(kedges) - 1 
    Nell = len(ells)
    
    # freq bin edges
    k2edges = kedges ** 2

    ksum = numpy.zeros(Nk)
    Nsum = numpy.zeros(Nk)
    Psum = numpy.zeros((Nk, Nell))
    
    # los index
    los_index = 'xyz'.index(los)
    
    for row in range(len(pm.k[0])):
        # pickup the singular plane that is single counted (r2c transform)
        singular = pm.k[2][-1] == 0

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
        
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5
    
        # the k sum
        ksum += numpy.bincount(dig_k, weights=scratch.flat, minlength=Nk+2)[1:-1]
    
        # take the sum of weights
        scratch[...] = 1.0
        # the singular plane is down weighted by 0.5
        scratch[singular] = 0.5
        Nsum += numpy.bincount(dig_k, weights=scratch.flat, minlength=Nk+2)[1:-1]

        # take the sum of power
        scratch[...] = c1[row].real * c2[row].real + c1[row].imag * c2[row].imag
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5
        
        for iell, ell in enumerate(ells):
            weights = scratch * (2*ell+1) * legendre(ell)(mu)
            Psum[:,iell] += numpy.bincount(dig_k, weights=weights.flat, minlength=Nk+2)[1:-1]

    ksum = pm.comm.allreduce(ksum, MPI.SUM)
    Psum = pm.comm.allreduce(Psum, MPI.SUM)
    Nsum = pm.comm.allreduce(Nsum, MPI.SUM)

    # reshape and slice to remove out of bounds points
    with numpy.errstate(invalid='ignore'):
        power = Psum / Nsum[:,None]
        kmean = ksum / Nsum
        N = 2*Nsum # factor of 2 for modes with negative z not in r2c transform

    # each complex field has units of L^3, so power is L^6
    power *= pm.BoxSize.prod() 
    return kmean, power, N, kedges

