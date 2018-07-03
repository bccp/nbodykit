import mcfit
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.integrate import quad

from .linear import LinearPower

NUM_PTS = 1024
KMIN = 1e-5
KMAX = 1e2

def isiterable(obj):
    """Returns `True` if the given object is iterable."""
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def vectorize_if_needed(func, *x):
    """ Helper function to vectorize functions on array inputs"""
    if any(map(isiterable, x)):
        return numpy.vectorize(func)(*x)
    else:
        return func(*x)

class ZeldovichPower(object):
    """
    The matter power spectrum in the Zel'dovich approximation.

    Parameters
    ----------
    cosmo : :class:`Cosmology`
        the cosmology instance
    z : float
        the redshift of the power spectrum
    transfer : str, optional
        string specifying the transfer function to use for the linear
        power spectrum; one of 'CLASS', 'EisensteinHu', 'NoWiggleEisensteinHu'

    Attributes
    ----------
    cosmo : class:`Cosmology`, astropy.cosmology.FLRW
        the object giving the cosmological parameters
    sigma8 : float
        the z=0 amplitude of matter fluctuations
    redshift : float
        the redshift to compute the power at
    Plin : class:`LinearPower`
        the linear power spectrum class used to compute the Zel'dovich power
    nmax : int
        max order of integrals.
    """
    def __init__(self, cosmo, redshift, transfer='CLASS', nmax=32):

        # initialize the linear power
        self.Plin = LinearPower(cosmo, redshift, transfer=transfer)
        self.nmax = nmax

        self.cosmo = self.Plin.cosmo
        self._sigma8 = self.cosmo.sigma8
        self.redshift = redshift

        # use low-k approx below this value
        self._k0_low = 5e-3

        # store meta-data
        self._attrs = {}
        self._attrs.update(self.Plin.attrs)

    @property
    def attrs(self):
        """
        The meta-data dictionary
        """
        self._attrs['redshift'] = self.redshift
        self._attrs['sigma8'] = self.sigma8
        return self._attrs

    def _setup(self):
        r"""
        Internal function to compute the following quantities, needed in
        the Zel'dovich approximation:

        .. math::

            \sigma_v^2 = 1/(6\pi^2) \int dk P_L(k),
            X(q) = \int \frac{dk}{2\pi^2} P_L(k) \left[ \frac{2}{3} - 2\frac{j_1(kq)}{kq} \right],
            Y(q) = \int \frac{dk}{2\pi^2} P_L(k) \left[ -2j_0(kq) + 6\frac{j_1(kq)}{kq} \right].
        """
        # set up the k-grid for integrals
        k = numpy.logspace(numpy.log10(KMIN), numpy.log10(KMAX), NUM_PTS)
        Pk = self.Plin(k)

        # compute the I0, I1 integrals we need
        self._r, I0 = ZeldovichJ0(k)(Pk, extrap=True)
        _, I1 = ZeldovichJ1(k)(Pk, extrap=True)

        # compute the X(r), Y(r) integrals we need
        self._sigmasq = self.Plin.velocity_dispersion(kmin=1e-5, kmax=10., limit=500)**2
        self._X = -2.*I1 + 2 * self._sigmasq
        self._Y = -2.*I0 + 6.*I1

        # needed for the low-k approx
        self._Q3 = quad(lambda q: (self.Plin(q)/q)**2, 1e-6, 100.)[0]

    @property
    def redshift(self):
        """
        The redshift of the power spectrum
        """
        return self._z

    @redshift.setter
    def redshift(self, value):
        self._z = value
        self.Plin.redshift = value
        self._setup()

    @property
    def sigma8(self):
        """
        The amplitude of matter fluctuations at :math:`z=0`.
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, value):
        self._sigma8 = value
        self.Plin.sigma8 = value
        self._setup()

    def _low_k_approx(self, k):
        r"""
        Return the low-k approximation of the Zel'dovich power. This computes:

        .. math::
            P(k) = (1 - k^2 \sigma_v^2 + 1/2 k^4 \sigma_v^4) P_L(k) + 0.5 Q_3(k),

        where :math:`Q_3(k)` is defined as

        .. math::
            Q_3(k) = \frac{k^4}{10\pi^2} \int dq \frac{P^2_L(q)}{q^2}.
        """
        Q3 = 1./(10.*numpy.pi**2) * k**4 * self._Q3

        Plin = self.Plin(k)
        term1 = (1 - k**2 * self._sigmasq + 0.5 * k**4 * self._sigmasq**2) * Plin
        term2 = 0.5 * Q3
        return term1 + term2

    def __call__(self, k):
        r"""
        Return the Zel'dovich power in :math:`h^{-3} \mathrm{Mpc}^3 at
        :attr:`redshift` and ``k``, where ``k`` is in units of
        :math:`h \mathrm{Mpc}^{-1}`.

        Parameters
        ----------
        k : float, array_like
            the wavenumbers to evaluate the power at
        """
        def Pzel_at_k(ki):

            # return the low-k approximation
            if ki < self._k0_low:
                return self._low_k_approx(ki)

            # do the full integral
            Pzel = 0.0
            for n in range(0, self.nmax + 1):

                I = ZeldovichPowerIntegral(self._r, n)
                if n > 0:
                    f = (ki*self._Y)**n * numpy.exp(-0.5*ki**2 * (self._X + self._Y))
                else:
                    f = numpy.exp(-0.5*ki**2 * (self._X + self._Y)) - numpy.exp(-ki**2*self._sigmasq)

                kk, this_Pzel = I(f, extrap=False)
                Pzel += spline(kk, this_Pzel)(ki)

            return Pzel

        return vectorize_if_needed(Pzel_at_k, k)

class ZeldovichJ0(mcfit.mcfit):
    r"""
    An integral over :math:`j_0` needed to compute the Zeldovich power. The
    integral is given by:

    .. math::

        I_0(r) = \int \frac{dk}{2\pi^2} P_L(k) j_0(kr).
    """
    def __init__(self, k):
        self.l = 0
        UK = mcfit.kernels.Mellin_SphericalBesselJ(self.l)
        mcfit.mcfit.__init__(self, k, UK, q=1.0, lowring=False)

        # set pre and post factors
        self.prefac = k
        self.postfac = 1 / (2*numpy.pi)**1.5

class ZeldovichJ1(mcfit.mcfit):
    r"""
    An integral over :math:`j_1` needed to compute the Zeldovich power. The
    integral is given by:

    .. math::

        I_1(r) = \int \frac{dk}{2\pi^2} P_L(k) \frac{j_1(kr)}{kr}.
    """
    def __init__(self, k):
        self.l = 1
        UK = mcfit.kernels.Mellin_SphericalBesselJ(self.l)
        mcfit.mcfit.__init__(self, k, UK, q=0, lowring=False)

        # set pre and post factors
        self.prefac = 1.0
        self.postfac = 1 / (2*numpy.pi)**1.5 / self.y

class ZeldovichPowerIntegral(mcfit.mcfit):
    r"""
    The integral needed to evaluate the density auto spectrum in the
    Zel'dovich approximation.

    This evaluates:

    .. math::
        I(k, n) = 4\pi \int dr r^2 \mathrm{exp}\left[-0.5k^2(X(r) + Y(r)) \right]
                    \left (\frac{k Y(r)}{r} \right)^n j_n(kr).
    """
    def __init__(self, r, n):
        self.n = n
        UK = mcfit.kernels.Mellin_SphericalBesselJ(self.n)
        mcfit.mcfit.__init__(self, r, UK, q=1.5-n, lowring=True)

        # set pre and post factors
        self.prefac = r**(3-n)
        self.postfac = (2*numpy.pi)**1.5
