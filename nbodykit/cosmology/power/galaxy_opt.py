import numpy
from . import transfers
from ..cosmology import Cosmology
from .linear import LinearPower

class GalaxyPower(object):
    """
    An object to compute the galaxy/redshift power spectrum and related quantities,
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
    b0 : float
        the linear bias of the galaxy in a gaussian field
    fnl : float
        the non-gaussian parameter
    p : float
        p takes a between 1 and 1.6. 
        p=1 for objects populating a fair sample of all the halos
        p=1.6 for objects that populate only recently merged halos
    Omega_m : float
        the matter density parameter
    H0 : float
        the Hubble constant (in units of km/s Mpc/h)
    c : float
        speed of light (in units of km/s)
        

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
    
    def __init__(self, cosmo, redshift ,b0 ,fNL ,p ,Omega_m ,H0=73.8 ,c=3e5 ,transfer='CLASS'):
        from astropy.cosmology import FLRW

        # convert astropy
        if isinstance(cosmo, FLRW):
            from nbodykit.cosmology import Cosmology
            cosmo = Cosmology.from_astropy(cosmo)

        # store a copy of the cosmology
        self.cosmo = cosmo.clone()
        
        #get the linear bias,p,fNL
        self.b=b0
        self.p=p
        self.fnl=fNL
        self.omega_m=Omega_m
        self.H0=H0
        self.c=c

        self.transfer = transfer

        # set redshift
        self.redshift = redshift
        
    
    def corrected_bias(self, k):
        """
        Returns the total/corrected galaxy bias in a non-gaussian field at the redshift
        specified by :attr:`redshift`.
        
        Parameters
        ---------
        k : float, array_like
            the wavenumber in units of :math:`h Mpc^{-1}`

        Returns
        -------
        total_bias : float, array_like
            the corrected galaxy bias in a non-gaussian field
        
        """
        Plin=LinearPower(self.cosmo, self.redshift, transfer=self.transfer)
        Pk=Plin(k)
        crit_density=1.686
        Dz=1/(1+self.redshift)         
        del_b= 3*self.fnl*(self.b-self.p)* crit_density * self.omega_m/(k**2*Pk**0.5*Dz) * (self.H0/self.c)**2
                
        total_bias=self.b + del_b
        
        return total_bias

    def __call__(self, k):
        """
        Return the galaxy power spectrum in units of
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
            the galaxy power spectrum evaluated at ``k`` in units of
            :math:`h^{-3} \mathrm{Mpc}^3`
        """

        Plin=LinearPower(self.cosmo, self.redshift, transfer=self.transfer)
        Pk=Plin(k)
        total_bias=self.corrected_bias(k)      
        
        Pgal = Pk * total_bias**2
        
        return Pgal
