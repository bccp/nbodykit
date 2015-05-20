import numpy

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction
from mpi4py import MPI

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
 
    def AnisotropicCIC(complex, w):
        for wi in w:
            tmp = (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
            complex[:] /= tmp

    wout = numpy.empty(pm.Nmesh//2)
    psout = numpy.empty(pm.Nmesh//2)

    def PowerSpectrum(complex, w):
        comm = pm.comm

        wedges = numpy.linspace(0, numpy.pi, wout.size + 1, endpoint=True)
        wedges += binshift * wedges[1]

        w2edges = wedges ** 2

        wsum = numpy.zeros(len(psout))
        P = numpy.zeros(len(psout))
        N = numpy.zeros(len(psout))
        for row in range(complex.shape[0]):
            # pickup the singular plane that is single counted (r2c transform)
            singular = w[0][-1] == 0

            scratch = w[0][row] ** 2
            for wi in w[1:]:
                scratch = scratch + wi[0] ** 2

            # now scratch stores w ** 2
            dig = numpy.digitize(scratch.flat, w2edges)

            # take the sum of w
            scratch **= 0.5
            # the singular plane is down weighted by 0.5
            scratch[singular] *= 0.5

            wsum1 = numpy.bincount(dig, weights=scratch.flat, minlength=wout.size + 2)[1: -1]
            wsum += comm.allreduce(wsum1, MPI.SUM)

            # take the sum of weights
            scratch[...] = 1.0
            # the singular plane is down weighted by 0.5
            scratch[singular] = 0.5

            N1 = numpy.bincount(dig, weights=scratch.flat, minlength=wout.size + 2)[1: -1]
            N += comm.allreduce(N1, MPI.SUM)

            # take the sum of power
            numpy.abs(complex[row], out=scratch)
            scratch[...] **= 2.0
            # the singular plane is down weighted by 0.5
            scratch[singular] *= 0.5

            P1 = numpy.bincount(dig, weights=scratch.flat, minlength=wout.size + 2)[1: -1]
            P += comm.allreduce(P1, MPI.SUM)

            psout[:] = P / N 
            wout[:] = wsum / N


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

    return kout, psout
