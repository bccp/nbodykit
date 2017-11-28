import numpy

class HalofitPower(object):
    """
    Nonlinear power spectrum computed using HaloFit via CLASS.

    Parameters
    ----------
    cosmo : :class:`Cosmology`, astropy.cosmology.FLRW
        the Cosmology instance; astropy cosmology objects are automatically
        converted
    redshift : float
        the redshift of the power spectrum

    Attributes
    ----------
    cosmo : class:`Cosmology`
        the object giving the cosmological parameters
    sigma8 : float
        the z=0 amplitude of matter fluctuations
    redshift : float
        the redshift to compute the power at
    """
    def __init__(self, cosmo, redshift):
        from astropy.cosmology import FLRW

        # convert astropy
        if isinstance(cosmo, FLRW):
            from nbodykit.cosmology import Cosmology
            cosmo = Cosmology.from_astropy(cosmo)

        # internal cosmology clone with nonlinear enabled
        self.cosmo = cosmo.clone(nonlinear=True)
        self.redshift = redshift
        self._sigma8 = self.cosmo.sigma8

        # store meta-data
        self._attrs = {}
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
    def sigma8(self):
        """
        The amplitude of matter fluctuations at :math:`z=0`.
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, value):
        self._sigma8 = value
        self.cosmo = self.cosmo.match(sigma8=value)
        self._attrs['cosmo'] = dict(self.cosmo)

    def __call__(self, k):
        r"""
        Return the power in units of :math:`h^{-3} \mathrm{Mpc}^3`.

        Parameters
        ----------
        k : float, array_like
            the wavenumbers in units of :math:`h \mathrm{Mpc}^{-1}`

        Returns
        -------
        Pk : float, array_like
            the linear power spectrum evaluated at ``k`` in units of
            :math:`h^{-3} \mathrm{Mpc}^3`
        """
        k = numpy.asarray(k)
        if k.max() > self.cosmo.P_k_max:
            msg = "results can only be computed up to k=%.2e h/Mpc; " %self.cosmo.P_k_max
            msg += "try increasing the Cosmology parameter 'P_k_max'"
            raise ValueError(msg)

        kmin = self.cosmo.P_k_min
        inrange = k > 1.00001*kmin

        Pk = numpy.zeros_like(k)
        k_in = k[inrange]; k_out = k[~inrange]

        # nonlinear power in range
        Pk[inrange] = self.cosmo.get_pk(k=k_in, z=self.redshift)

        # linear power on large scales (small k)
        Pk[~inrange] = self.cosmo.get_pklin(k=k_out, z=self.redshift)
        return Pk
