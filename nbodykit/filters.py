from nbodykit.base.mesh import MeshFilter

import numpy

class TopHat(MeshFilter):
    """ A TopHat filter defined in Fourier space.

        Notes
        -----
        A fourier space filter is different from a configuration space
        filter. The TopHat in fourier space creates ringing effects
        due to the truncation / discretization of modes.

    """
    kind = 'wavenumber'
    mode = 'complex'

    def __init__(self, r):
        """
            Parameters
            ----------
            r : float
                radius of the TopHat filter
        """
        self.r = r

    def filter(self, k, v):
        r = self.r
        k = sum(ki ** 2 for ki in k) ** 0.5
        kr = k * r
        w = 3 * (numpy.sin(kr) / kr **3 - numpy.cos(kr) / kr ** 2)
        w[k == 0] = 1.0
        return w * v

class Gaussian(MeshFilter):
    """ A gaussian filter

        .. math ::

            G(r) = exp(-0.5 k^2 r^2)

    """
    kind = 'wavenumber'
    mode = 'complex'

    def __init__(self, r):
        """
            Parameters
            ----------
            r : float
                radius of the Gaussian filter
        """
        self.r = r

    def filter(self, k, v):
        r = self.r
        k2 = sum(ki ** 2 for ki in k)
        return numpy.exp(- 0.5 * k2 * r**2)

