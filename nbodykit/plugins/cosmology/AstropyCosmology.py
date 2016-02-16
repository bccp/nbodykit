from nbodykit.extensionpoints import Cosmology

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
    
class AstropyCosmology(Cosmology):
    plugin_name = 'AstropyCosmology'
    
    @classmethod
    def register(kls):
        
        p = kls.parser
        p.add_argument('-H0', type=float, default=67.6,
            help='the Hubble constant at z=0, in km/s/Mpc; default: 67.6')
        p.add_argument('-Om0', type=float, default=0.31,
            help='matter density/critical density at z=0; default: 0.31')
        p.add_argument('-Ode0', type=float, default=0.69,
            help='dark energy density/critical density at z=0; default: 0.69')
        p.add_argument('-w0', type=float, default=-1.,
            help='dark energy equation of state; default: -1')
        p.add_argument('-Tcmb0', type=float, default=2.7255,
            help='temperature of the CMB in K at z=0; default: 2.7255')
        p.add_argument('-Neff', type=float, default=3.04,
            help='effective number of neutrino species; default 3.04')
        p.add_argument('-m_nu', nargs='+', type=neutrino_mass, 
            default=0.,
            help='mass of neutrino species in eV; default: 0.')
        p.add_argument('-flat', action='store_true', default=False,
            help='if `True`, automatically sets `Ode0` such that `Ok0` is zero')
    
    def finalize_attributes(self):
        """
        Finalize the attributes by initialize the cosmology instance based
        on the input parameters
        """
        from astropy import cosmology, units
        
        # convert neutrino mass to a astropy `Quantity`
        self.m_nu = units.Quantity(self.m_nu, 'eV')
        
        # initialize the cosmology object
        kw = {k:getattr(self,k) for k in ['w0', 'Tcmb0', 'Neff', 'm_nu']}
        if not self.flat:
            self.cosmo = cosmology.wCDM(self.H0, self.Om0, self.Ode0, **kw)
        else:
            self.cosmo = cosmology.FlatwCDM(self.H0, self.Om0, **kw)
    
    def comoving_distance(self, z):
        """
        Returns the comoving distance to z in units of `Mpc/h`
        """
        return self.cosmo.comoving_distance(z).value * self.cosmo.h