import numpy
from . import transfers
from ..cosmology import Cosmology

class LinearPower(object):
    """
    An object to compute the linear power spectrum and related quantities,
    using a transfer function from the CLASS code or the analytic
    Eisenstein & Hu approximation.

    Parameters
    ----------
    cosmo : :class:`Cosmology`, astropy.cosmology.FLRW
        the cosmology instance; astropy cosmology objects are automatically
        converted to the proper type
    redshift : float
        the redshift of the power spectrum
    transfer : str, optional
        string specifying the transfer function to use; one of
        'CLASS', 'EisensteinHu', 'NoWiggleEisensteinHu'

    Attributes
    ----------
    cosmo : class:`Cosmology`
        the object giving the cosmological parameters
    sigma8 : float
        the z=0 amplitude of matter fluctuations
    redshift : float
        the redshift to compute the power at
    transfer : str
        the type of transfer function used
    """
    def __init__(self, cosmo, redshift, transfer='CLASS'):
        from astropy.cosmology import FLRW

        # convert astropy
        if isinstance(cosmo, FLRW):
            from nbodykit.cosmology import Cosmology
            cosmo = Cosmology.from_astropy(cosmo)

        # store a copy of the cosmology
        self.cosmo = cosmo.clone()

        # set sigma8 to the cosmology value
        self._sigma8 = self.cosmo.sigma8

        # setup the transfers
        if transfer not in transfers.available:
            raise ValueError("'transfer' should be one of %s" %str(transfers.available))
        self.transfer = transfer

        # initialize internal transfers
        c = self.cosmo.clone() # transfers get an internal copy
        self._transfer = getattr(transfers, transfer)(c, redshift)
        self._fallback = transfers.EisensteinHu(c, redshift) # fallback to analytic when out of range

        # normalize to proper sigma8
        self._norm = 1.
        self.redshift = 0;
        self._norm = (self._sigma8 / self.sigma_r(8.))**2 # sigma_r(z=0, r=8)

        # set redshift
        self.redshift = redshift

        # store meta-data
        self._attrs = {}
        self._attrs['transfer'] = transfer
        self._attrs['cosmo'] = dict(cosmo)

    @property
    def attrs(self):
        """
        The meta-data dictionary
        """
        self._attrs['redshift'] = self.redshift
        self._attrs['sigma8'] = self.sigma8
        return self._attrs

    @property
    def redshift(self):
        """
        The redshift of the power spectrum
        """
        return self._z

    @redshift.setter
    def redshift(self, value):
        self._z = value
        self._transfer.redshift = value
        self._fallback.redshift = value

    @property
    def sigma8(self):
        """
        The present day value of ``sigma_r(r=8 Mpc/h)``, used to normalize
        the power spectrum, which is proportional to the square of this value.

        The power spectrum can re-normalized by setting a different
        value for this parameter
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, value):
        """
        Set the sigma8 value and normalize the power spectrum to the new value
        """
        # re-scale the normalization
        self._norm *= (value / self._sigma8)**2

        # update to this sigma8
        self._sigma8 = value

    def __call__(self, k):
        """
        Return the linear power spectrum in units of
        :math:`h^{-3} \mathrm{Mpc}^3` at the redshift specified by
        :attr:`redshift`.

        The transfer function used to evaluate the power spectrum is
        specified by the ``transfer`` attribute.

        Parameters
        ---------
        k : float, array_like
            the wavenumber in units of :math:`h Mpc^{-1}`

        Returns
        -------
        Pk : float, array_like
            the linear power spectrum evaluated at ``k`` in units of
            :math:`h^{-3} \mathrm{Mpc}^3`
        """
        if self.transfer != "CLASS":
            Pk = k**self.cosmo.n_s * self._transfer(k)**2
        else:
            k = numpy.asarray(k)
            kmax = self.cosmo.P_k_max
            inrange = k < 0.99999*kmax # prevents rounding errors

            # the return array (could be scalar array)
            Pk = numpy.zeros_like(k)

            # k values in and out of valid range
            k_in = k[inrange]; k_out = k[~inrange]

            # use CLASS in range
            Pk[inrange] = k_in**self.cosmo.n_s * self._transfer(k_in)**2

            # use Eisentein-Hu out of range
            if len(k_out):
                analytic_Tk = self._fallback(k_out)
                analytic_Tk *= self._transfer(kmax)/ self._fallback(kmax)
                Pk[~inrange] = k_out**self.cosmo.n_s * analytic_Tk**2

        return self._norm * Pk

    def velocity_dispersion(self, kmin=1e-5, kmax=10., **kwargs):
        r"""
        The velocity dispersion in units of of :math:`\mathrm{Mpc/h}` at
        ``redshift``.

        This returns :math:`\sigma_v`, defined as

        .. math::

            \sigma_v^2 = \frac{1}{3} \int_a^b \frac{d^3 q}{(2\pi)^3} \frac{P(q,z)}{q^2}.

        Parameters
        ----------
        kmin : float, optional
            the lower bound for the integral, in units of :math:`\mathrm{Mpc/h}`
        kmax : float, optional
            the upper bound for the integral, in units of :math:`\mathrm{Mpc/h}`
        """
        from scipy.integrate import quad

        def integrand(logq):
            q = numpy.exp(logq)
            return q*self(q)
        sigmasq = quad(integrand, numpy.log(kmin), numpy.log(kmax), **kwargs)[0] / (6*numpy.pi**2)
        return sigmasq**0.5

    def sigma_r(self, r, kmin=1e-5, kmax=1e1):
        r"""
        The mass fluctuation within a sphere of radius ``r``, in
        units of :math:`h^{-1} Mpc` at ``redshift``.

        This returns :math:`\sigma`, where

        .. math::

            \sigma^2 = \int_0^\infty \frac{k^3 P(k,z)}{2\pi^2} W^2_T(kr) \frac{dk}{k},

        where :math:`W_T(x) = 3/x^3 (\mathrm{sin}x - x\mathrm{cos}x)` is
        a top-hat filter in Fourier space.

        The value of this function with ``r=8`` returns
        :attr:`sigma8`, within numerical precision.

        Parameters
        ----------
        r : float, array_like
            the scale to compute the mass fluctation over, in units of
            :math:`h^{-1} Mpc`
        kmin : float, optional
            the lower bound for the integral, in units of :math:`\mathrm{Mpc/h}`
        kmax : float, optional
            the upper bound for the integral, in units of :math:`\mathrm{Mpc/h}`
        """
        import mcfit
        from scipy.interpolate import InterpolatedUnivariateSpline as spline

        k = numpy.logspace(numpy.log10(kmin), numpy.log10(kmax), 1024)
        Pk = self(k)
        R, sigmasq = mcfit.TophatVar(k)(Pk)

        return spline(R, sigmasq)(r)**0.5


def NoWiggleEHPower(cosmo, redshift):
    import warnings
    warnings.warn("NoWiggleEHPower is deprecated. Use LinearPower with transfer set to 'NoWiggleEisensteinHu'", FutureWarning)
    return LinearPower(cosmo=cosmo, redshift=redshift, transfer="NoWiggleEisensteinHu")

def EHPower(cosmo, redshift):
    import warnings
    warnings.warn("NoWiggleEHPower is deprecated. Use LinearPower with transfer set to 'EisensteinHu'", FutureWarning)
    return LinearPower(cosmo=cosmo, redshift=redshift, transfer="EisensteinHu")
