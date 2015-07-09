import numpy

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction
from mpi4py import MPI

def measure2Dpower(pm, c1, c2, Nmu, binshift=0.0, shotnoise=0.0, los='z'):
    """ Measure 2D power spectrum P(k,mu) from density field painted on pm 

        The power spectrum is measured in bins of k and mu. The k bins extend 
        from 0 to the Nyquist (Nmesh / 2), with the units consistent with 
        the units of the BoxSize. The mu range extends from 0 to 1.
        

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
        
        los : str
            the line-of-sight direction, which the angle `mu` is defined with
            respect to. Default is `z` (last column)
            
    """
    Nfreq = pm.Nmesh//2
    ndims = (Nfreq+2, Nmu+2)
    
    wedges = numpy.linspace(0, numpy.pi, Nfreq + 1, endpoint=True)
    wedges += binshift * wedges[1]
    # mu bin edges
    muedges = numpy.linspace(0, 1, Nmu+1, endpoint=True)
    
    # freq bin edges
    w2edges = wedges ** 2

    musum = numpy.zeros(ndims)
    wsum = numpy.zeros(ndims)
    Psum = numpy.zeros(ndims)
    Nsum = numpy.zeros(ndims)
    
    # los index
    los_index = 'xyz'.index(los)
    
    for row in range(len(pm.w[0])):
        # pickup the singular plane that is single counted (r2c transform)
        singular = pm.w[2][-1] == 0

        # now scratch stores w ** 2
        scratch = numpy.float64(pm.w[0][row] ** 2)
        for wi in pm.w[1:]:
            scratch = scratch + wi[0] ** 2

        if len(scratch.flat) == 0:
            # no data
            continue

        dig_w = numpy.digitize(scratch.flat, w2edges)
    
        # make scratch just w
        scratch **= 0.5
    
        # store mu
        with numpy.errstate(invalid='ignore'):
            if los_index == 0:
                mu = abs(pm.w[los_index][row]/scratch)
            else:
                mu = abs(pm.w[los_index][0]/scratch)
        dig_mu = numpy.digitize(mu.flat, muedges)
        
        # make the multi-index
        multi_index = numpy.ravel_multi_index([dig_w, dig_mu], ndims)
    
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5
        mu[singular] *= 0.5
    
        # the w sum
        wsum.flat += numpy.bincount(multi_index, weights=scratch.flat, minlength=wsum.size)
    
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

    wsum = pm.comm.allreduce(wsum, MPI.SUM)
    musum = pm.comm.allreduce(musum, MPI.SUM)
    Psum = pm.comm.allreduce(Psum, MPI.SUM)
    Nsum = pm.comm.allreduce(Nsum, MPI.SUM)

    # reshape and slice to remove out of bounds points
    with numpy.errstate(invalid='ignore'):
        power = (Psum / Nsum)[1:-1, 1:-1]
        wmean = (wsum / Nsum)[1:-1, 1:-1]
        mumean = (musum / Nsum)[1:-1, 1:-1]
        N = Nsum[1:-1, 1:-1]

    # measure the raw power spectrum, nothing is removed.
    kout = wmean * pm.Nmesh / pm.BoxSize
    kedges = wedges * pm.Nmesh / pm.BoxSize
    power *= (pm.BoxSize) ** 3
    power -= shotnoise

    return kout, mumean, power, N, [kedges, muedges]

