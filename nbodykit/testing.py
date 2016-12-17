import numpy

def PowerlawPowerSpectrum(alpha=-2.0, P1=300.):

    def PowerSpectrum(k):
        k[k==0] = 1
        return P1 * k**alpha

    PowerSpectrum.attrs = {}
    PowerSpectrum.attrs['alpha'] = alpha
    PowerSpectrum.attrs['P1'] = P1
    PowerSpectrum.attrs['type'] = 'powerlaw'

    return PowerSpectrum

TestingPowerSpectrum = PowerlawPowerSpectrum()
