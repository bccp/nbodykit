from nbodykit.base.catalog import CatalogSource, column
from nbodykit import transform
import numpy

class HaloCatalog(CatalogSource):
    """
    A wrapper Source class to interface nicely with 
    :class:`halotools.sim_manager.UserSuppliedHaloCatalog`
    """
    def __init__(self, source, cosmo, redshift, mdef='vir',
                    mass='Mass', velocity='Velocity', position='Position', 
                     use_cache=False, comm=None):
        """
        Parameters
        ----------
        source : CatalogSource
            the source holding the particles to be interpreted as halos
        particle_mass : float
            the 
        cosmo : nbodykit.cosmology.Cosmology
            the cosmology instance; 
        redshift : float
            the redshift of the halo catalog
        mdef : str; optional
            string specifying mass definition, used for computing default
            halo radii and concentration; should be 'vir' or 'XXXc' or 
            'XXXm' where 'XXX' is an int specifying the overdensity
        mass : str; optional
            the column name specifying the mass of each halo
        position : str; optional
            the column name specifying the position of each halo
        velocity : str; optional
            the column name specifying the velocity of each halo
        """
        if comm is None:
            comm = source.comm
            
        self._source = source
        self.cosmo = cosmo
        
        # get the attrs from the source
        self.attrs.update(source.attrs)
        
        # and save the parameters
        self.attrs['redshift'] = redshift
        self.attrs['mass']     = mass
        self.attrs['velocity'] = velocity
        self.attrs['position'] = position
        self.attrs['mdef']     = mdef
                
        # init the base class
        CatalogSource.__init__(self, comm=comm, use_cache=use_cache)
        
    @property
    def size(self):
        return self._source.size
    
    @column
    def Mass(self):
        return self.make_column(self._source[self.attrs['mass']])
        
    @column
    def Position(self):
        return self.make_column(self._source[self.attrs['position']])
        
    @column
    def Velocity(self):
        return self.make_column(self._source[self.attrs['velocity']])
    
    @column
    def VelocityOffset(self):
        """
        This multiplies Velocity by 1 / (a*100*E(z)) = 1 / (a H(z)/h)
        """
        z = self.attrs['redshift']
        rsd_factor = (1+z) / (100*self.cosmo.efunc(z))
        return self['Velocity'] * rsd_factor
        
    @column
    def Concentration(self):
        z = self.attrs['redshift']
        mdef = self.attrs['mdef']
        return transform.HaloConcentration(self['Mass'], self.cosmo, z, mdef=mdef)
        
    @column
    def Radius(self):
        z = self.attrs['redshift']
        mdef = self.attrs['mdef']
        return transform.HaloRadius(self['Mass'], self.cosmo, z, mdef=mdef)
        
    def to_halotools(self, BoxSize=None, selection='Selection'):
        """
        Return the source as a :class:`halotools.sim_manager.UserSuppliedHaloCatalog`.
        The Halotools catalog only holds the local data, although halos are labeled
        via the ``halo_id`` column with the global index 
        
        
        Parameters
        ----------
        BoxSize : float, array_like; optional
            the size of the box; note that anisotropic boxes are currently
            not supported by halotools
        selection : str; optional
            the name of the column to slice the data on before converting
            to a halotools catalog
        
        Returns
        -------
        cat : :class:`halotools.sim_manager.UserSuppliedHaloCatalog`
            the Halotools halo catalog, storing the local halo data
        """
        from halotools.sim_manager import UserSuppliedHaloCatalog
        from halotools.empirical_models import model_defaults
        
        # make sure we have a BoxSize
        if BoxSize is None:
            BoxSize = self.attrs.get('BoxSize', None)
        if BoxSize is None:
            raise ValueError("please specify a 'BoxSize' to convert to a halotools catalog")
        
        # anisotropic boxes not yet supported
        if not numpy.isscalar(BoxSize):
            BoxSize = numpy.asarray(BoxSize)
            if not (BoxSize == BoxSize[0]).all():
                raise ValueError("halotools does not currently support anisotropic boxes; see astropy/halotools#641")
            else:
                BoxSize = BoxSize[0]
                
        assert selection in self, "'%s' selection column is not valid" %selection
                
        # compute the columns
        sel = self.compute(self[selection])
        cols = ['Position', 'Velocity', 'Mass', 'Radius', 'Concentration']
        cols = self.compute(*[self[col][sel] for col in cols])
        Position, Velocity, Mass, Radius, Concen = [col for col in cols]
        
        # names of the mass and radius fields, based on mass def
        mkey = model_defaults.get_halo_mass_key(self.attrs['mdef'])
        rkey = model_defaults.get_halo_boundary_key(self.attrs['mdef'])
        
        # global halo ids (across all ranks)
        sizes = self.comm.allgather(self.size)
        start = sum(sizes[:self.comm.rank])
        halo_id = numpy.arange(start, start+self.size, dtype='i8')[sel]
        
        # data columns
        kws                  = {}
        kws['halo_x']        = Position[:,0]
        kws['halo_y']        = Position[:,1]
        kws['halo_z']        = Position[:,2]
        kws['halo_vx']       = Velocity[:,0]
        kws['halo_vy']       = Velocity[:,1]
        kws['halo_vz']       = Velocity[:,2]
        kws[mkey]            = Mass
        kws[rkey]            = Radius
        kws['halo_nfw_conc'] = Concen
        kws['halo_id']       = halo_id
        kws['halo_upid']     = numpy.zeros(len(Position)) - 1
        kws['halo_local_id'] = numpy.arange(0, self.size, dtype='i8')[sel]            
        
        # add metadata too
        kws['cosmology']     = self.cosmo
        kws['redshift']      = self.attrs['redshift']
        kws['Lbox']          = BoxSize
        kws['particle_mass'] = self.attrs.get('particle_mass', 1.0)
        kws['mdef']          = self.attrs['mdef']
        
        return UserSuppliedHaloCatalog(**kws)
        
    
    
