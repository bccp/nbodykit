import numpy as np
import abc
from scipy.integrate import simps
from nbodykit.extern.six import add_metaclass

@add_metaclass(abc.ABCMeta)
class LinearPowerBase(object):
    """
    Class for computing linear power spectra from a transfer function
    """
    def __init__(self, cosmo, redshift):
        """
        Parameters
        ----------
        cosmo : nbodykit.cosmology.Cosmology
            the cosmology instance; must have 'Ob0', 'n_s', and 'sigma8'
            parameters
        redshift : float
            the redshift to compute the power spectrum at
        """
        # useful message if astropy cosmology is passed
        from astropy.cosmology import FLRW
        if isinstance(cosmo, FLRW):
            raise ValueError("please provide a nbodykit.cosmology.Cosmology class; see Cosmology.from_astropy")
        self.cosmo = cosmo

        # check required parameters
        required = ['Ob0', 'sigma8', 'n_s']
        for param in required:
            if param not in self.cosmo:
                raise ValueError("'%s' attribute of cosmology must be provided" %param)

        # set sigma8 to the cosmology value
        self._sigma8 = self._sigma8_0 = self.cosmo.sigma8

        # redshift
        self.redshift = redshift
        self.D = self.cosmo.growth_function(self.redshift)

    @property
    def attrs(self):
        """
        Cosmology parameters stored in `attrs` dictionary
        """
        from astropy.units import Quantity
        attrs = {'cosmo.%s' %k : self.cosmo[k] for k in self.cosmo}

        # current value of sigma8
        attrs['cosmo.sigma8'] = self.sigma8

        return attrs

    @abc.abstractmethod
    def transfer(self, k):
        """
        Subclasses should override this to return the transfer function
        """
        return NotImplemented

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
        self._sigma8 = value
        self._normalize()

    def _normalize(self):
        """
        Internal function that is called when ``sigma8`` is set, ensuring
        the integral over the power spectrum gives the proper ``sigma8``
        value
        """
        # set to unity so we get the un-normalized sigma
        self._norm = 1.0

        # power is proportional so square of sigma8
        self._norm = (self.sigma8/self.sigma_r(8.))**2

    def __call__(self, k, no_BAO=False):
        """
        Return the linear power spectrum in units of [Mpc/h]^3
        at the redshift :attr:`z`

        The amplitude scales with redshift through the square of the
        growth function, evaluated using ``cosmo.growth_function``

        Paramters
        ---------
        k : float, array_like
            the wavenumbers in units of h/Mpc

        Returns
        -------
        Plin : float, array_like
            the linear power spectrum evaluated at k in [Mpc/h]^3
        """
        if not hasattr(self, '_norm'): self._normalize()
        if np.isscalar(k) and k == 0:
            return 1.
        k = np.asarray(k)
        T = self.transfer(k)
        r = self.D**2 * (self.sigma8/self._sigma8_0)**2 * self._norm * T ** 2 * k ** self.cosmo.n_s
        r = np.asarray(r)
        r[k == 0] = 1.
        return r

    def sigma_r(self, r):
        """
        The mass fluctuation within a sphere of radius ``R``, in
        units of Mpc/h

        The value of this function with ``R=8`` returns
        :attr:`sigma8`, within numerical precision

        Parameters
        ----------
        r : float
            the scale to compute the mass fluctation over, in
            units of `Mpc/h`
        """
        if not hasattr(self, '_norm'): self._normalize()

        def integrand(lnk):
            k = np.exp(lnk)
            kr = k*r
            j1_kr = np.sin(kr)/kr**3 - np.cos(kr)/kr**2
            T = self.transfer(k)
            return k*k*k * (k ** self.cosmo.n_s) * (T ** 2) * (3*j1_kr) ** 2

        norm = self._norm * (self.sigma8 / self._sigma8_0)**2
        lnk = np.log(np.logspace(-4, 4, 4096))
        sigma_sq = norm*simps(integrand(lnk), x=lnk) / (2*np.pi**2)
        return sigma_sq**0.5


class EHPower(LinearPowerBase):
    """
    Eisenstein & Hu (1998) fitting function with BAO wiggles

    From EH 1998, Eqs. 26,28-31.
    """
    def __init__(self, cosmo, redshift):
        """
        Parameters
        ----------
        cosmo : nbodykit.cosmology.Cosmology
            the cosmology instance; must have 'Ob0', 'n_s', and 'sigma8'
            parameters
        redshift : float
            the redshift to compute the power spectrum at
        """
        LinearPowerBase.__init__(self, cosmo, redshift)
        self._set_params()

    def _set_params(self):
        """
        Initialize the parameters of the fitting formula
        """
        self.Obh2      = self.cosmo.Ob0 * self.cosmo.h ** 2
        self.Omh2      = self.cosmo.Om0 * self.cosmo.h ** 2
        self.f_baryon  = self.cosmo.Ob0 / self.cosmo.Om0
        self.theta_cmb = self.cosmo.Tcmb0 / 2.7

        # redshift and wavenumber of equality
        self.z_eq = 2.5e4 * self.Omh2 * self.theta_cmb ** (-4) # this is 1 + z
        self.k_eq = 0.0746 * self.Omh2 * self.theta_cmb ** (-2) # units of 1/Mpc

        # sound horizon and k_silk
        self.z_drag_b1 = 0.313 * self.Omh2 ** -0.419 * (1 + 0.607 * self.Omh2 ** 0.674)
        self.z_drag_b2 = 0.238 * self.Omh2 ** 0.223
        self.z_drag    = 1291 * self.Omh2 ** 0.251 / (1. + 0.659 * self.Omh2 ** 0.828) * \
                           (1. + self.z_drag_b1 * self.Obh2 ** self.z_drag_b2)

        self.r_drag = 31.5 * self.Obh2 * self.theta_cmb ** -4 * (1000. / (1+self.z_drag))
        self.r_eq   = 31.5 * self.Obh2 * self.theta_cmb ** -4 * (1000. / self.z_eq)

        self.sound_horizon = 2. / (3.*self.k_eq) * np.sqrt(6. / self.r_eq) * \
                    np.log((np.sqrt(1 + self.r_drag) + np.sqrt(self.r_drag + self.r_eq)) / (1 + np.sqrt(self.r_eq)) )
        self.k_silk = 1.6 * self.Obh2 ** 0.52 * self.Omh2 ** 0.73 * (1 + (10.4*self.Omh2) ** -0.95)

        # alpha_c
        alpha_c_a1 = (46.9*self.Omh2) ** 0.670 * (1 + (32.1*self.Omh2) ** -0.532)
        alpha_c_a2 = (12.0*self.Omh2) ** 0.424 * (1 + (45.0*self.Omh2) ** -0.582)
        self.alpha_c = alpha_c_a1 ** -self.f_baryon * alpha_c_a2 ** (-self.f_baryon**3)

        # beta_c
        beta_c_b1 = 0.944 / (1 + (458*self.Omh2) ** -0.708)
        beta_c_b2 = 0.395 * self.Omh2 ** -0.0266
        self.beta_c = 1. / (1 + beta_c_b1 * ((1-self.f_baryon) ** beta_c_b2) - 1)

        y = self.z_eq / (1 + self.z_drag)
        alpha_b_G = y * (-6.*np.sqrt(1+y) + (2. + 3.*y) * np.log((np.sqrt(1+y)+1) / (np.sqrt(1+y)-1)))
        self.alpha_b = 2.07 *  self.k_eq * self.sound_horizon * (1+self.r_drag)**-0.75 * alpha_b_G

        self.beta_node = 8.41 * self.Omh2 ** 0.435
        self.beta_b    = 0.5 + self.f_baryon + (3. - 2.*self.f_baryon) * np.sqrt( (17.2*self.Omh2) ** 2 + 1 )

    def transfer(self, k):
        """
        Return the transfer function with BAO wiggles

        This is normalized to unity on large scales

        Paramters
        ---------
        k : float, array_like
            the wavenumbers in units of h/Mpc
        """
        if np.isscalar(k) and k == 0.:
            return 1.0

        k = np.asarray(k)
        # only compute k > 0 modes
        valid = k > 0.

        k = k[valid] * self.cosmo.h # now in 1/Mpc
        q = k / (13.41*self.k_eq)
        ks = k*self.sound_horizon

        T_c_ln_beta   = np.log(np.e + 1.8*self.beta_c*q)
        T_c_ln_nobeta = np.log(np.e + 1.8*q);
        T_c_C_alpha   = 14.2 / self.alpha_c + 386. / (1 + 69.9 * q ** 1.08)
        T_c_C_noalpha = 14.2 + 386. / (1 + 69.9 * q ** 1.08)

        T_c_f = 1. / (1. + (ks/5.4) ** 4)
        f = lambda a, b : a / (a + b*q**2)
        T_c = T_c_f * f(T_c_ln_beta, T_c_C_noalpha) + (1-T_c_f) * f(T_c_ln_beta, T_c_C_alpha)

        s_tilde = self.sound_horizon * (1 + (self.beta_node/ks)**3) ** (-1./3.)
        ks_tilde = k*s_tilde

        T_b_T0 = f(T_c_ln_nobeta, T_c_C_noalpha)
        T_b_1 = T_b_T0 / (1 + (ks/5.2)**2 )
        T_b_2 = self.alpha_b / (1 + (self.beta_b/ks)**3 ) * np.exp(-(k/self.k_silk) ** 1.4)
        T_b = np.sinc(ks_tilde/np.pi) * (T_b_1 + T_b_2)

        T = np.ones(valid.shape)
        T[valid] = self.f_baryon*T_b + (1-self.f_baryon)*T_c;
        return T

class NoWiggleEHPower(LinearPowerBase):
    """
    Eisenstein & Hu (1998) fitting function without BAO wiggles

    From EH 1998, Eqs. 26,28-31.
    """
    def __init__(self, cosmo, redshift):
        """
        Parameters
        ----------
        cosmo : nbodykit.cosmology.Cosmology
            the cosmology instance; must have 'Ob0', 'n_s', and 'sigma8'
            parameters
        redshift : float
            the redshift to compute the power spectrum at
        """
        LinearPowerBase.__init__(self, cosmo, redshift)
        self._set_params()

    def _set_params(self):
        """
        Initialize the parameters of the fitting formula
        """
        self.Obh2      = self.cosmo.Ob0 * self.cosmo.h ** 2
        self.Omh2      = self.cosmo.Om0 * self.cosmo.h ** 2
        self.f_baryon  = self.cosmo.Ob0 / self.cosmo.Om0
        self.theta_cmb = self.cosmo.Tcmb0 / 2.7

        # wavenumber of equality
        self.k_eq = 0.0746 * self.Omh2 * self.theta_cmb ** (-2) # units of 1/Mpc

        self.sound_horizon = self.cosmo.h * 44.5 * np.log(9.83/self.Omh2) / np.sqrt(1 + 10 * self.Obh2** 0.75) # in Mpc/h
        self.alpha_gamma = 1 - 0.328 * np.log(431*self.Omh2) * self.f_baryon + 0.38* np.log(22.3*self.Omh2) * self.f_baryon ** 2


    def transfer(self, k):
        """
        Return the transfer function without BAO wiggles

        This is normalized to unity on large scales

        Paramters
        ---------
        k : float, array_like
            the wavenumbers in units of h/Mpc
        """
        if np.isscalar(k) and k == 0.:
            return 1.0

        # only compute k > 0 modes
        k = np.asarray(k)
        valid = k > 0.

        k = k[valid] * self.cosmo.h # in 1/Mpc now
        ks = k * self.sound_horizon / self.cosmo.h
        q = k / (13.41*self.k_eq)

        gamma_eff = self.Omh2 * (self.alpha_gamma + (1 - self.alpha_gamma) / (1 + (0.43*ks) ** 4))
        q_eff = q * self.Omh2 / gamma_eff
        L0 = np.log(2*np.e + 1.8 * q_eff)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)

        T = np.ones(valid.shape)
        T[valid] = L0 / (L0 + C0 * q_eff**2)
        return T
