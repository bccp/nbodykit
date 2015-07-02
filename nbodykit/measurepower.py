import numpy

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction
from mpi4py import MPI

def AnisotropicCIC(comm, complex, w):
    for wi in w:
        tmp = (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
        complex[:] /= tmp

def measurepower(pm, complex, binshift=0.0, remove_cic="anisotropic", shotnoise=0.0):
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
        remove_cic : string
            can be "anisotropic", "isotropic", and "none".
            the default is anisotropic which is the proper way to remove cloud in cell
            kernel
        shotnoise : 0.0
            remove the shot noise contribution from the final power spectrum
            The shot noise is (pm.BoxSize) ** 3 / Ntot, where Ntot is the total
            number of particles.
            
        Notes
        -----
        After the measurement the density field is destroyed.

    """
    pm.complex[:] = complex
 
    wout = numpy.empty(pm.Nmesh//2)
    psout = numpy.empty(pm.Nmesh//2)
    Nout = numpy.empty(pm.Nmesh//2)
    
    wedges = numpy.linspace(0, numpy.pi, wout.size + 1, endpoint=True)
    wedges += binshift * wedges[1]

    def PowerSpectrum(comm, complex, w):

        w2edges = wedges ** 2

        wsum = numpy.zeros(len(psout))
        P = numpy.zeros(len(psout))
        N = numpy.zeros(len(psout))
        for row in range(complex.shape[0]):
            # pickup the singular plane that is single counted (r2c transform)
            singular = w[2][-1] == 0

            scratch = numpy.float64(w[0][row] ** 2)
            for wi in w[1:]:
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
            numpy.abs(complex[row], out=scratch)
            scratch[...] **= 2.0
            # the singular plane is down weighted by 0.5
            scratch[singular] *= 0.5

            P1 = numpy.bincount(dig, weights=scratch.flat, minlength=wout.size + 2)[1: -1]
            P += P1

        wsum = comm.allreduce(wsum, MPI.SUM)
        P = comm.allreduce(P, MPI.SUM)
        N = comm.allreduce(N, MPI.SUM)

        psout[:] = P / N 
        wout[:] = wsum / N
        Nout[:] = N

    chain = [
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
    ]

    if remove_cic == 'anisotropic':
        chain.append(AnisotropicCIC)

    chain.append(PowerSpectrum)
        
    # measure the raw power spectrum, nothing is removed.
    pm.transfer(chain)
    kout = wout * pm.Nmesh / pm.BoxSize
    psout *= (pm.BoxSize) ** 3

    if remove_cic == 'isotropic':
        tmp = 1.0 - 0.666666667 * numpy.sin(wout * 0.5) ** 2
        psout /= tmp

    psout -= shotnoise

    return kout, psout, Nout, wedges*pm.Nmesh/pm.BoxSize
    
def measure2Dpower(pm, complex, binshift=0.0, remove_cic="anisotropic", shotnoise=0.0, Nmu=5):
    """ Measure 2D power spectrum P(k,mu) from density field painted on pm 

        The power spectrum is measured in bins of k and mu. The k bins extend 
        from 0 to the Nyquist (Nmesh / 2), with the units consistent with 
        the units of the BoxSize. The mu range extends from 0 to 1.

        Parameters
        ----------
        pm : ParticleMesh
            A particle mesh object

        complex: array_like
            the complex fourier space field to measure power from.

        binshift   : float
            shift the center of bins by this fraction if a bin width
        remove_cic : string
            can be "anisotropic", "isotropic", and "none".
            the default is anisotropic which is the proper way to remove cloud in cell
            kernel
        shotnoise : 0.0
            remove the shot noise contribution from the final power spectrum
            The shot noise is (pm.BoxSize) ** 3 / Ntot, where Ntot is the total
            number of particles.
        Nmu : int
            The number of mu bins to use when binning in the power spectrum. 
            
        Notes
        -----
        After the measurement the density field is destroyed.

    """
    Nfreq = pm.Nmesh//2
    
    pm.complex[:] = complex
    
    class PowerSpectrum2D(object):
    
        def __init__(self, k_bins, mu_bins):
            
            # Nmesh//2 is number of freq bins, Nmu is number of mu bins
            self.mu    = numpy.empty((k_bins, mu_bins))
            self.w     = numpy.empty((k_bins, mu_bins))
            self.power = numpy.empty((k_bins, mu_bins))
            self.N     = numpy.empty((k_bins, mu_bins))
            
            self.Nmu = mu_bins
            self.Nfreq = k_bins

            self.wedges = numpy.linspace(0, numpy.pi, self.Nfreq + 1, endpoint=True)
            self.wedges += binshift * self.wedges[1]
            # mu bin edges
            self.muedges = numpy.linspace(0, 1, self.Nmu+1, endpoint=True)
            
        def __call__(self, comm, complex, w):
            # freq bin edges
            w2edges = self.wedges ** 2
        
        
            ndims = (self.Nfreq+2, self.Nmu+2)
            musum = numpy.zeros(ndims[0]*ndims[1])
            wsum = numpy.zeros(ndims[0]*ndims[1])
            P = numpy.zeros(ndims[0]*ndims[1])
            N = numpy.zeros(ndims[0]*ndims[1])
            
            for row in range(complex.shape[0]):
                # pickup the singular plane that is single counted (r2c transform)
                singular = w[2][-1] == 0

                scratch = numpy.float64(w[0][row] ** 2)
                for wi in w[1:]:
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
                    mu = abs(w[-1][0]/scratch)
                dig_mu = numpy.digitize(mu.flat, self.muedges)
            
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

                N1 = numpy.bincount(multi_index, weights=scratch.flat, minlength=ndims[0]*ndims[1])
                N += N1

                # take the sum of power
                numpy.abs(complex[row], out=scratch)
                scratch[...] **= 2.0
                # the singular plane is down weighted by 0.5
                scratch[singular] *= 0.5

                P1 = numpy.bincount(multi_index, weights=scratch.flat, minlength=ndims[0]*ndims[1])
                P += P1

            wsum = comm.allreduce(wsum, MPI.SUM)
            musum = comm.allreduce(musum, MPI.SUM)
            P = comm.allreduce(P, MPI.SUM)
            N = comm.allreduce(N, MPI.SUM)

            # reshape and slice to remove out of bounds points
            with numpy.errstate(invalid='ignore'):
                self.power[:] = (P / N).reshape(ndims)[1:-1, 1:-1]
                self.w[:] = (wsum / N).reshape(ndims)[1:-1, 1:-1]
                self.mu[:] = (musum / N).reshape(ndims)[1:-1, 1:-1]
                self.N[:] = N.reshape(ndims)[1:-1, 1:-1]

    chain = [
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
    ]
    if remove_cic == 'anisotropic':
        chain.append(AnisotropicCIC)

    P = PowerSpectrum2D(Nfreq, Nmu)
    chain.append(P)
        
    # measure the raw power spectrum, nothing is removed.
    pm.transfer(chain)
    kout = P.w * pm.Nmesh / pm.BoxSize
    kedges = P.wedges * pm.Nmesh / pm.BoxSize
    P.power *= (pm.BoxSize) ** 3

    if remove_cic == 'isotropic':
        tmp = 1.0 - 0.666666667 * numpy.sin(wout * 0.5) ** 2
        P.power /= tmp

    P.power -= shotnoise

    return kout, P.mu, P.power, P.N, [kedges, P.muedges]

