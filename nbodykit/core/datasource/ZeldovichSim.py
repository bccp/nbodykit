from nbodykit.core import DataSource
from nbodykit import utils
import numpy

         
class ZeldovichSimDataSource(DataSource):
    """
    DataSource to return a catalog of simulated objects,  
    using the Zel'dovich displacement field and overdensity field
    generated from an input power spectrum
    
    .. note::
    
        This class requires `classylss` to be installed; 
        see https://pypi.python.org/pypi/classylss
    """
    plugin_name = "ZeldovichSim"
    
    def __init__(self, nbar, redshift, BoxSize, Nmesh, bias=2., rsd=None, seed=None):        
        
        self.nbar     = nbar
        self.redshift = redshift
        self.BoxSize  = BoxSize
        self.Nmesh    = Nmesh
        self.bias     = bias
        self.rsd      = rsd
        self.seed     = seed
        
        # create the local random seed from the global seed and comm size
        self.local_seed = utils.local_random_seed(self.seed, self.comm)
        
        # crash if no cosmology provided
        if self.cosmo is None:
            raise ValueError("a cosmology must be specified via the `cosmo` keyword to use %s" %self.plugin_name)
            
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "simulated particles using the Zel'dovich approximation"
        
        # positional
        s.add_argument('nbar', type=float,
            help='the desired number density of the catalog in the box')
        s.add_argument("redshift", type=float,
            help='the desired redshift of the catalog')
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("Nmesh", type=int, 
            help='the number of cells per box side in the gridded mesh')
        
        # optional
        s.add_argument("seed", type=int,
            help='the number used to seed the random number generator')
        s.add_argument("rsd", type=str, choices="xyz",
            help="the direction to add redshift space distortions to; default is no RSD")
        s.add_argument("bias", type=float,
            help="the linear bias factor to apply")
    
    def parallel_read(self, columns, full=False):
        """
        Return the position of the simulated particles -- 'Position' is the 
        only valid column
        """   
        # classylss is required to call CLASS and create a power spectrum
        try: import classylss
        except: raise ImportError("`classylss` is required to use %s" %self.plugin_name)
        
        # the other imports
        from nbodykit import mockmaker
        from pmesh.pm import ParticleMesh
        from astropy.utils.misc import NumpyRNGContext
        
        # initialize the CLASS parameters 
        pars = classylss.ClassParams.from_astropy(self.cosmo.engine)

        try:
            cosmo = classylss.Cosmology(pars)
        except Exception as e:
            raise ValueError("error running CLASS for the specified cosmology: %s" %str(e))
        
        # initialize the linear power spectrum object
        Plin = classylss.power.LinearPS(cosmo, z=self.redshift)
        
        # the particle mesh for gridding purposes
        pm = ParticleMesh(BoxSize=self.BoxSize, Nmesh=[self.Nmesh]*3, dtype='f4', comm=self.comm)
        
        # generate initialize fields and Poisson sample with fixed local seed
        with NumpyRNGContext(self.local_seed):
        
            # compute the linear overdensity and displacement fields
            delta, disp = mockmaker.gaussian_real_fields(pm, Plin, compute_displacement=True)
        
            # sample to Poisson points
            f = cosmo.f_z(self.redshift) # growth rate to do RSD in the Zel'dovich approx
            kws = {'rsd':self.rsd, 'f':f, 'bias':self.bias}
            pos = mockmaker.poisson_sample_to_points(delta, disp, pm, self.nbar, **kws)

        # yield position
        yield [pos if col == 'Position' else None for col in columns]


