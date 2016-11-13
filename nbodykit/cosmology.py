from nbodykit.plugins.schema import attribute
from astropy import cosmology, units

class CosmologyBase(object):
    """
    Base class for computing cosmology-dependent quantities, possibly
    sampling them at at set of points and returning interpolated results 
    (for speed purposes)
    """
    class sampled_function:
        """
        Class to represent a "sampled" version of a function
        """
        def __init__(self, func, x, *args, **kwargs):
            from scipy import interpolate
            self.func = func
            self.x = x

            # assign function name and docs
            self.__name__ = None
            self.__doc__  = None
            if func.__name__ != None:
                self.__name__ = self.func.__name__ + " [Sampled to %i pts]" %len(x)
            if func.__doc__ != None:
                self.__doc__ = "sampled function : \n\n" + self.func.__doc__

            self.spline = interpolate.InterpolatedUnivariateSpline(x, func(x,*args,**kwargs))

        def __call__(self, y):
            return self.spline(y)
            
    def sample(self, methodname, x, *args, **kwargs):
        if not hasattr(self, methodname):
            raise ValueError("no such method '%s' to sample" %methodname)

        # unsample first
        if self.is_sampled(methodname):
            self.unsample(methodname)
        
        # set the sampled function
        tmp = getattr(self, methodname)
        setattr(self, methodname, self.sampled_function(tmp, x, *args, **kwargs))

    def unsample(self,methodname):
        if self.is_sampled(methodname):
            tmp = getattr(self, methodname).func
            setattr(self, methodname, tmp)
        else:
            raise ValueError("cannot unsample %s" %methodname)
        
    def is_sampled(self,methodname):
        return getattr(self,methodname).__class__ == self.sampled_function
                
def neutrino_mass(value):
    """
    Function to cast an input string or list to a `astropy.units.Quantity`,
    with units of `eV` to represent neutrino mass
    """    
    if isinstance(value, str):
        value = [float(i) for i in value.split()]
    
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        elif len(value) != 3:
            raise ValueError("either a single neutrino mass, or a mass for each of the 3 species must be provided")
    return value
    
class Cosmology(CosmologyBase):
    """
    A class for computing cosmology-dependent quantites, which uses
    `astropy.Cosmology` to do the calculations
    
    Notes
    -----
    *   this class supports the :class:`~astropy.cosmology.LambdaCDM` and 
        :class:`~astropy.cosmology.wCDM` classes from `astropy` (and their
        flat equivalents)
    *   additions to the fiducial `LCDM` model include the dark energy equation 
        of state `w0` and massive neutrinos
    *   if `flat = True`, the dark energy density is set automatically
    *   the underlying astropy class is stored as the `engine` attribute
    """  
    @attribute('flat', type=bool, help="if `True`, automatically set `Ode0` such that `Ok0` is zero")
    @attribute('m_nu', type=neutrino_mass, help="mass of neutrino species in eV")
    @attribute('Neff', type=float, help="effective number of neutrino species")
    @attribute('Tcmb0', type=float, help="temperature of the CMB in K at z=0")
    @attribute('w0', type=float, help="dark energy equation of state")
    @attribute('Ode0', type=float, help="dark energy density/critical density at z=0")
    @attribute('Ob0', type=float, help="baryon density/critical density at z=0")
    @attribute('Om0', type=float, help="matter density/critical density at z=0")
    @attribute('H0', type=float, help="the Hubble constant at z=0, in km/s/Mpc")
    def __init__(self, H0=67.6, Om0=0.31, Ob0=0.0486, Ode0=0.69, w0=-1., Tcmb0=2.7255, 
                    Neff=3.04, m_nu=0., flat=False):

        # set the parameters
        self.H0   = H0
        self.Om0  = Om0
        self.Ob0  = Ob0
        self.Ode0 = Ode0
        self.w0 = w0
        self.Tcmb0 = Tcmb0
        self.Neff = Neff
        self.flat = flat
        
        # convert neutrino mass to a astropy `Quantity`
        self.m_nu = units.Quantity(m_nu, 'eV')
        
        # the astropy keywords
        kws = {k:getattr(self,k) for k in ['w0', 'Tcmb0', 'Neff', 'm_nu', 'Ob0']}
        
        # determine the astropy class
        if self.w0 == -1.0: # cosmological constant
            cls = 'LambdaCDM'
            kws.pop('w0')
        else:
            cls = 'wCDM'
        if self.flat: cls = 'Flat' + cls
        
        # initialize the cosmology object
        if not self.flat:
            self.engine = getattr(cosmology, cls)(self.H0, self.Om0, self.Ode0, **kws)
        else:
            self.engine = getattr(cosmology, cls)(self.H0, self.Om0, **kws)
            self.Ode0 = self.engine.Ode0 # set this automatically
    
    def comoving_distance(self, z):
        """
        Returns the comoving distance to z in units of `Mpc/h`
        """
        
        return self.engine.comoving_distance(z).value * self.engine.h
