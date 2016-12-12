from nbodykit.base.grid import GridSource
from nbodykit import CurrentMPIComm, mockmaker
from nbodykit.utils import MPINumpyRNGContext

class LinearGrid(GridSource):
    """
    A source to generate a ``RealField`` grid directly from the 
    linear power spectrum, using a specified cosmology and redshift
    
    The linear power spectrum is computed using :mod:`classylss`, 
    which is a python wrapper around CLASS 
    """
    @CurrentMPIComm.enable
    def __init__(self, cosmo, redshift, BoxSize, seed=None, comm=None):
        """
        Parameters
        ----------
        cosmo : subclass of astropy.cosmology.FLRW
           the cosmology used to generate the linear power spectrum (using CLASS) 
        redshift : float
            the redshift of the linear power spectrum to generate
        BoxSize : float, 3-vector of floats
            the size of the box to generate the grid on
        seed : int, optional
            the global random seed, used to set the seeds across all ranks
        comm : MPI communicator
            the MPI communicator
        """        
        # classylss is required to call CLASS and create a power spectrum
        try: import classylss
        except: raise ImportError("`classylss` is required to use %s" %self.__class__.__name__)
        
        # initialize the CLASS parameters and save dict version
        self.pars = classylss.ClassParams.from_astropy(cosmo)
        self.attrs['cosmo'] = dict(self.pars)
        
        # cosmology and communicator
        self.comm    = comm
        self.cosmo   = cosmo
        
        # save the rest of the attributes as meta-data
        self.attrs['redshift'] = redshift
        self.attrs['seed']     = seed
        self.attrs['BoxSize']  = BoxSize
        
        GridSource.__init__(self, comm)
            
    def paint(self, pm):
        """
        Load a grid from file, and paint to the ParticleMesh represented by ``pm``
        
        Parameters
        ----------
        pm : pmesh.pm.ParticleMesh
            the particle mesh object to which we will paint the grid
        
        Returns
        -------
        real : pmesh.pm.RealField
            an array-like object holding the interpolated grid
        """
        import classylss
        
        # run CLASS backend for the specified cosmo
        try:
            cosmo = classylss.Cosmology(self.pars)
        except Exception as e:
            raise ValueError("error running CLASS for the specified cosmology: %s" %str(e))
    
        # initialize the linear power spectrum object
        Plin = classylss.power.LinearPS(cosmo, z=self.attrs['redshift'])
        
        # generate linear density field with desired seed
        with MPINumpyRNGContext(self.attrs['seed'], self.comm):
            real, _ = mockmaker.gaussian_real_fields(pm, Plin, compute_displacement=False)
    
        # apply transformations to real field
        self.painter.transform(self, real)
        return real
