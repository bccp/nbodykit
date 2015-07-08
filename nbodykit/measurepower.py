import numpy

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction
from mpi4py import MPI

def measurepower(pm, c1, c2, binshift=0.0, shotnoise=0.0):
    """ Measure power spectrum from density field painted on pm 

        The power spectrum is measured in bins of k, from 0 to the Nyquist (Nmesh / 2)
        The units is consistent with the units of the BoxSize.

        Parameters
        ----------
        pm : ParticleMesh
            A particle mesh object

        complex: array_like
            the complex fourier space field to measure power from.

        binshift   : float
            shift the center of bins by this fraction if a bin width

        shotnoise : 0.0
            remove the shot noise contribution from the final power spectrum
            The shot noise is (pm.BoxSize) ** 3 / Ntot, where Ntot is the total
            number of particles.
            
        Notes
        -----
        After the measurement the density field is destroyed.

    """
 
    wout = numpy.empty(pm.Nmesh//2)
    psout = numpy.empty(pm.Nmesh//2)
    Nout = numpy.empty(pm.Nmesh//2)
    
    wedges = numpy.linspace(0, numpy.pi, wout.size + 1, endpoint=True)
    wedges += binshift * wedges[1]

    w2edges = wedges ** 2

    wsum = numpy.zeros(len(psout))
    P = numpy.zeros(len(psout))
    N = numpy.zeros(len(psout))
    for row in range(len(pm.w[0])):
        # pickup the singular plane that is single counted (r2c transform)
        singular = pm.w[2][-1] == 0

        scratch = numpy.float64(pm.w[0][row] ** 2)
        for wi in pm.w[1:]:
            scratch = scratch + wi[0] ** 2

        # now scratch stores w ** 2
        if len(scratch.flat) == 0:
            # no data
            continue

        dig = numpy.digitize(scratch.flat, w2edges)

        # take the sum of w
        scratch **= 0.5
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5

        wsum1 = numpy.bincount(dig, weights=scratch.flat, minlength=wout.size + 2)[1: -1]
        wsum += wsum1

        # take the sum of weights
        scratch[...] = 1.0
        # the singular plane is down weighted by 0.5
        scratch[singular] = 0.5

        N1 = numpy.bincount(dig, weights=scratch.flat, minlength=wout.size + 2)[1: -1]
        N += N1

        # take the sum of power
        scratch[...] = c1[row].real * c2[row].real + c1[row].imag * c2[row].imag
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5

        P1 = numpy.bincount(dig, weights=scratch.flat, minlength=wout.size + 2)[1: -1]
        P += P1

    wsum = pm.comm.allreduce(wsum, MPI.SUM)
    P = pm.comm.allreduce(P, MPI.SUM)
    N = pm.comm.allreduce(N, MPI.SUM)

    psout[:] = P / N 
    wout[:] = wsum / N
    Nout[:] = N

    kout = wout * pm.Nmesh / pm.BoxSize
    psout *= (pm.BoxSize) ** 3

    psout -= shotnoise

    return kout, psout, Nout, wedges*pm.Nmesh/pm.BoxSize
    
def measure2Dpower(pm, c1, c2, binshift=0.0, shotnoise=0.0, Nmu=5):
    """ Measure 2D power spectrum P(k,mu) from density field painted on pm 

        The power spectrum is measured in bins of k and mu. The k bins extend 
        from 0 to the Nyquist (Nmesh / 2), with the units consistent with 
        the units of the BoxSize. The mu range extends from 0 to 1.

        Parameters
        ----------
        pm : ParticleMesh
            A particle mesh object

        c1: array_like
            the complex fourier space field to measure power from.

        c2: array_like
            the complex fourier space field to measure power from.

        binshift   : float
            shift the center of bins by this fraction if a bin width

        shotnoise : 0.0
            remove the shot noise contribution from the final power spectrum
            The shot noise is (pm.BoxSize) ** 3 / Ntot, where Ntot is the total
            number of particles.

        Nmu : int
            The number of mu bins to use when binning in the power spectrum. 
            
    """
    Nfreq = pm.Nmesh//2
    
    # Nmesh//2 is number of freq bins, Nmu is number of mu bins
    mumean    = numpy.empty((Nfreq, Nmu))
    wmean     = numpy.empty((Nfreq, Nmu))
    power = numpy.empty((Nfreq, Nmu))
    N     = numpy.empty((Nfreq, Nmu))
    
    wedges = numpy.linspace(0, numpy.pi, Nfreq + 1, endpoint=True)
    wedges += binshift * wedges[1]
    # mu bin edges
    muedges = numpy.linspace(0, 1, Nmu+1, endpoint=True)
    
    # freq bin edges
    w2edges = wedges ** 2

    ndims = (Nfreq+2, Nmu+2)
    musum = numpy.zeros(ndims[0]*ndims[1])
    wsum = numpy.zeros(ndims[0]*ndims[1])
    Psum = numpy.zeros(ndims[0]*ndims[1])
    Nsum = numpy.zeros(ndims[0]*ndims[1])
    
    for row in range(len(pm.w[0])):
        # pickup the singular plane that is single counted (r2c transform)
        singular = pm.w[2][-1] == 0

        scratch = numpy.float64(pm.w[0][row] ** 2)
        for wi in pm.w[1:]:
            scratch = scratch + wi[0] ** 2

        # now scratch stores w ** 2

        if len(scratch.flat) == 0:
            # no data
            continue

        dig_w = numpy.digitize(scratch.flat, w2edges)
    
        # make scratch just w
        scratch **= 0.5
    
        # store mu
        with numpy.errstate(invalid='ignore'):
            mu = abs(pm.w[-1][0]/scratch)
        dig_mu = numpy.digitize(mu.flat, muedges)
    
        # make the multi-index
        multi_index = numpy.ravel_multi_index([dig_w, dig_mu], ndims)
    
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5
        mu[singular] *= 0.5
    
        # the w sum
        wsum1 = numpy.bincount(multi_index, weights=scratch.flat, minlength=ndims[0]*ndims[1])
        wsum += wsum1
    
        # the mu sum
        musum1 = numpy.bincount(multi_index, weights=mu.flat, minlength=ndims[0]*ndims[1])
        musum += musum1
    
        # take the sum of weights
        scratch[...] = 1.0
        # the singular plane is down weighted by 0.5
        scratch[singular] = 0.5

        Nsum += numpy.bincount(multi_index, weights=scratch.flat, minlength=ndims[0]*ndims[1])

        # take the sum of power
        scratch[...] = c1[row].real * c2[row].real + c1[row].imag * c2[row].imag
        # the singular plane is down weighted by 0.5
        scratch[singular] *= 0.5

        Psum += numpy.bincount(multi_index, weights=scratch.flat, minlength=ndims[0]*ndims[1])

    wsum = pm.comm.allreduce(wsum, MPI.SUM)
    musum = pm.comm.allreduce(musum, MPI.SUM)
    Psum = pm.comm.allreduce(Psum, MPI.SUM)
    Nsum = pm.comm.allreduce(Nsum, MPI.SUM)

    # reshape and slice to remove out of bounds points
    with numpy.errstate(invalid='ignore'):
        power[:] = (Psum / Nsum).reshape(ndims)[1:-1, 1:-1]
        wmean[:] = (wsum / Nsum).reshape(ndims)[1:-1, 1:-1]
        mumean[:] = (musum / Nsum).reshape(ndims)[1:-1, 1:-1]
        N[:] = Nsum.reshape(ndims)[1:-1, 1:-1]

    # measure the raw power spectrum, nothing is removed.

    kout = wmean * pm.Nmesh / pm.BoxSize
    kedges = wedges * pm.Nmesh / pm.BoxSize
    power *= (pm.BoxSize) ** 3

    power -= shotnoise

    return kout, mumean, power, N, [kedges, muedges]

