import numpy
import mcfit
from scipy.interpolate import InterpolatedUnivariateSpline
from .power.zeldovich import ZeldovichPower

NUM_PTS = 1024

def xi_to_pk(r, xi, ell=0, extrap=False):
    r"""
    Return a callable function returning the power spectrum multipole of degree
    :math:`\ell`, as computed from the Fourier transform of the input :math:`r`
    and :math:`\xi_\ell(r)` arrays.

    This uses the :mod:`mcfit` package perform the FFT.

    Parameters
    ----------
    r : array_like
        separation values where ``xi`` is evaluated
    xi : array_like
        the array holding the correlation function multipole values
    ell : int
        multipole degree of the input correlation function and the output power
        spectrum; monopole by default
    extrap : bool, optional
        whether to extrapolate the power spectrum with a power law; can improve
        the smoothness of the FFT

    Returns
    -------
    InterpolatedUnivariateSpline :
        a spline holding the interpolated power spectrum values
    """
    P = mcfit.xi2P(r, l=ell)
    kk, Pk = P(xi, extrap=extrap)
    return InterpolatedUnivariateSpline(kk, Pk)


def pk_to_xi(k, Pk, ell=0, extrap=True):
    r"""
    Return a callable function returning the correlation function multipole of
    degree :math:`\ell`, as computed from the Fourier transform of the input
    :math:`k` and :math:`P_\ell(k)` arrays.

    This uses the :mod:`mcfit` package perform the FFT.

    Parameters
    ----------
    k : array_like
        wavenumbers where ``Pk`` is evaluated
    Pk : array_like
        the array holding the power spectrum multipole values
    ell : int
        multipole degree of the input power spectrum and the output correlation
        function; monopole by default
    extrap : bool, optional
        whether to extrapolate the power spectrum with a power law; can improve
        the smoothness of the FFT

    Returns
    -------
    InterpolatedUnivariateSpline :
        a spline holding the interpolated correlation function values
    """
    xi = mcfit.P2xi(k, l=ell)
    rr, CF = xi(Pk, extrap=extrap)
    return InterpolatedUnivariateSpline(rr, CF)


class CorrelationFunction(object):
    """
    Evaluate the correlation function by Fourier transforming
    a power spectrum object, with automatic re-scaling with redshift and sigma8.

    Parameters
    ----------
    power : callable
         a callable power spectrum that returns the power at a given ``k``;
         this should have ``redshift``, ``sigma8``, and ``cosmo`` attributes
    """
    def __init__(self, power):

        self.power = power

        # check required attributes
        for attr in ['redshift', 'sigma8', 'cosmo']:
            if not hasattr(power, attr):
                raise AttributeError("input power spectrum object must have a ``%s`` attribute" %attr)

        # meta-data
        self._attrs = {}
        self._attrs.update(getattr(self.power, 'attrs', {}))

    @property
    def attrs(self):
        self._attrs['redshift'] = self.redshift
        self._attrs['sigma8'] = self.sigma8
        return self._attrs

    @property
    def redshift(self):
        return self.power.redshift

    @redshift.setter
    def redshift(self, value):
        self.power.redshift = value

    @property
    def sigma8(self):
        return self.power.sigma8

    @sigma8.setter
    def sigma8(self, value):
        self.power.sigma8 = value

    @property
    def cosmo(self):
        return self.power.cosmo

    def __call__(self, r, smoothing=0., kmin=1e-5, kmax=10.):
        """
        Return the correlation function (dimensionless) for separations ``r``

        Parameters
        ----------
        r : float, array_like
            the separation array in units of :math:`h^{-1} \mathrm(Mpc)`
        smoothing  : float, optional
            the std deviation of the Fourier space Gaussian smoothing to apply
            to P(k) before taking the FFT
        kmin : float, optional
            the minimum ``k`` value to compute P(k) for before taking the FFT
        kmax : float, optional
            the maximum ``k`` value to compute P(k) for before taking the FFT
        """
        k = numpy.logspace(numpy.log10(kmin), numpy.log10(kmax), NUM_PTS)

        # power with smoothing
        Pk = self.power(k)
        Pk *= numpy.exp(-(k*smoothing)**2)

        # only extrap if not zeldovich
        extrap = not isinstance(self.power, ZeldovichPower)
        return pk_to_xi(k, Pk, extrap=extrap)(r)
