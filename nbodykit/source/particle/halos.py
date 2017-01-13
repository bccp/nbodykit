from nbodykit.base.particles import ParticleSource, column
import numpy
from halotools.empirical_models import profile_helpers, ConcMass

class HaloCatalog(ParticleSource):
    """
    A wrapper Source class to interface nicely with 
    :class:`halotools.sim_manager.UserSuppliedHaloCatalog`
    """
    def __init__(self, source, particle_mass, cosmo, redshift, 
                    length='Length', velocity='Velocity', position='Position', 
                    mdef='vir', use_cache=False):
        """
        Parameters
        ----------
        source : ParticleSource
            the source holding the particles to be interpreted as halos
        particle_mass : float
            the 
        cosmo : nbodykit.cosmology.Cosmology
            the cosmology instance; 
        redshift : float
            the redshift of the halo catalog
        length : str; optional
            the column name specifying the "length" of each halo, which is the 
            number of particles per halo; halo mass is computed as the
            particle mass times the length
        velocity : str; optional
            the column name specifying the velocity of each halo
        position : str; optional
            the column name specifying the position of each halo
        mdef : str; optional
            string specifying mass definition, used for computing default
            halo radii and concentration; should be 'vir' or 'XXXc' or 
            'XXXm' where 'XXX' is an int specifying the overdensity
        use_cache : bool; optional
            use dask to cache the intermediate the columns            
        """
        self._source = source
        self.cosmo = cosmo
        
        self.attrs['particle_mass']   = particle_mass
        self.attrs['redshift']        = redshift
        self.attrs['length']          = length
        self.attrs['velocity']        = velocity
        self.attrs['position']        = position
        self.attrs['mdef']            = mdef
        
        # default c(M) follows Eqs 12 & 13 of Dutton & Maccio 2014, arXiv:1402.7073
        self._conc_model = ConcMass(cosmology=cosmo.engine, conc_mass_model='dutton_maccio14',
                                    mdef=mdef, redshift=redshift)
        
        
        ParticleSource.__init__(self, comm=source.comm, use_cache=use_cache)
        
    @property
    def size(self):
        return self._source.size
    
    @column
    def HaloMass(self):
        mass = self.attrs['particle_mass'] * self._source[self.attrs['length']]
        return self.make_column(mass)
        
    @column
    def HaloPosition(self):
        return self.make_column(self._source[self.attrs['position']])
        
    @column
    def HaloVelocity(self):
        return self.make_column(self._source[self.attrs['velocity']])
        
        
    @column
    def HaloConcentration(self):
        conc = self._conc_model.compute_concentration(prim_haloprop=self['HaloMass'])
        return self.make_column(conc)
        
    @column
    def HaloRadius(self):
        kws = {'mass':self['HaloMass'], 'cosmology':self.cosmo.engine, 
               'redshift':self.attrs['redshift'], 'mdef':self.attrs['mdef']
              }
        return self.make_column(profile_helpers.halo_mass_to_halo_radius(**kws))
        
    def to_halotools(self, BoxSize, selection='Selection', extra={}):
        """
        Return the source as a :class:`halotools.sim_manager.UserSuppliedHaloCatalog`.
        The Halotools catalog only holds the local data (this is not a collective
        operation).
        
        By default, the halotools catalog will included the data from all columns
        in :attr:`columns`. Additional columns can be included by providing
        the mapping between nbodykit and halotool column names 
        
        Parameters
        ----------
        BoxSize : float, array_like
            the size of the box; note that anisotropic boxes are currently
            not supported by halotools
        selection : str; optional
            the name of the column to slice the data on before converting
            to a halotools catalog
        extra : dict, optional
            a mapping between columns in the Source (keys) and the corresponding
            columns in the halotools (values) for any extra columns the user
            wishes to include
        
        Returns
        -------
        cat : :class:`halotools.sim_manager.UserSuppliedHaloCatalog`
            the Halotools halo catalog, storing the local halo data
        """
        from halotools.sim_manager import UserSuppliedHaloCatalog
        from halotools.empirical_models import model_defaults
        
        if not numpy.isscalar(BoxSize):
            BoxSize = numpy.asarray(BoxSize)
            if not (BoxSize == BoxSize[0]).all():
                raise ValueError("halotools does not currently support anisotropic boxes; see astropy/halotools#641")
            else:
                BoxSize = BoxSize[0]
                
        assert selection in self, "'%s' selection column is not valid" %selection
                
        # compute the columns
        sel = self.compute(self[selection])
        cols = ['HaloPosition', 'HaloVelocity', 'HaloMass', 'HaloRadius', 'HaloConcentration']
        cols = self.compute(*[self[col][sel] for col in cols])
        Position, Velocity, Mass, Radius, Concen = [col for col in cols]
        
        # names of the mass and radius fields, based on mass def
        mkey = model_defaults.get_halo_mass_key(self.attrs['mdef'])
        rkey = model_defaults.get_halo_boundary_key(self.attrs['mdef'])
        
        kws                  = {}
        kws['redshift']      = self.attrs['redshift']
        kws['Lbox']          = BoxSize
        kws['particle_mass'] = self.attrs['particle_mass']
        kws['halo_x']        = Position[:,0]
        kws['halo_y']        = Position[:,1]
        kws['halo_z']        = Position[:,2]
        kws['halo_vx']       = Velocity[:,0]
        kws['halo_vy']       = Velocity[:,1]
        kws['halo_vz']       = Velocity[:,2]
        kws[mkey]            = Mass
        kws[rkey]            = Radius
        kws['halo_nfw_conc'] = Concen
        kws['halo_id']       = numpy.arange(len(Position))
        kws['halo_upid']     = numpy.zeros(len(Position)) - 1
        return UserSuppliedHaloCatalog(**kws)
        
    
    
