from nbodykit.core import DataSource
import numpy


def set_random_seed(f, seed):
    """
    Decorator to seed the random seed in the 
    ``mc_occuputation_*`` functions of the HOD model instance
    """
    def wrapper(*args, **kwargs):
        kwargs['seed'] = seed
        return f(*args, **kwargs)
    
    for a in f.__dict__:
        setattr(wrapper, a, f.__dict__[a])
    return wrapper
    
    
def model_with_random_seed(model, seed):
    """
    Update the relevant functions of the model to use the
    specified random seed
    """
    # update the satellite profile functions
    sat_functions = ['mc_pos', 'mc_radial_velocity', 'mc_solid_sphere']
    sat_prof = model.model_dictionary['satellites_profile']
    for sat_func in sat_functions:
        f = set_random_seed(getattr(sat_prof, sat_func), seed)
        setattr(sat_prof, sat_func, f)
        
    # update the occupation functions
    occup_functions = ['mc_occupation_centrals', 'mc_occupation_satellites']
    for occup_func in occup_functions:
        f = set_random_seed(getattr(model, occup_func), seed)
        setattr(model, occup_func, f)


class Zheng07HodDataSource(DataSource):
    """
    A `DataSource` that uses the Hod prescription of 
    Zheng et al. 2007 to populate an input halo catalog with galaxies, 
    and returns the (Position, Velocity) of those galaxies
    
    The mock population is done using `halotools` (http://halotools.readthedocs.org)
    The Hod model is of the commonly-used form:
    
    * logMmin: Minimum mass required for a halo to host a central galaxy
    * sigma_logM: Rate of transition from <Ncen>=0 --> <Ncen>=1
    * alpha: Power law slope of the relation between halo mass and <Nsat>
    * logM0: Low-mass cutoff in <Nsat>
    * logM1: Characteristic halo mass where <Nsat> begins to assume a power law form
    
    See the documentation for the `halotools` builtin Zheng07 Hod model, 
    for further details regarding the Hod 
    """
    plugin_name = "Zheng07Hod"
    
    def __init__(self, halocat, redshift, logMmin=13.031, sigma_logM=0.38, 
                    alpha=0.76, logM0=13.27, logM1=14.08, rsd=None, seed=None):
        """
        Default values for Hod values from Reid et al. 2014
        """
        from nbodykit.distributedarray import GatherArray
        
        self.halocat    = halocat
        self.redshift   = redshift 
        self.logMmin    = logMmin
        self.sigma_logM = sigma_logM
        self.alpha      = alpha
        self.logM0      = logM0
        self.logM1      = logM1
        self.rsd        = rsd
        self.seed       = seed
        
        
        # load halotools
        try:
            from halotools import sim_manager
        except:
            raise ValueError("`halotools` must be installed to use '%s' DataSource" %self.plugin_name)
            
        # need cosmology
        if self.cosmo is None:
            raise AttributeError("a cosmology instance is required to populate a Hod")
        
        # set global redshift and cosmology as global defaults
        sim_manager.sim_defaults.default_cosmology = self.cosmo.engine
        sim_manager.sim_defaults.default_redshift = self.redshift
        from halotools import empirical_models as em_models
        
        # grab the halocat BoxSize
        self.BoxSize = self.halocat.BoxSize
        
        # read data from halo catalog and then gather to root
        columns = ['Position','Velocity', 'Mass']
        try:
            with self.halocat.open() as stream:
                [data] = stream.read(columns, full=True)
        except Exception as e:
            m = "error reading from halo catalog in %s; " %self.plugin_name
            m += "be sure that the following columns are supported: %s\n" %str(columns)
            raise ValueError(m + "original exception: %s" %str(e))
        alldata = [GatherArray(d, self.comm, root=0) for d in data]
        
        # rank 0 does the populating
        if self.comm.rank == 0:
            Position, Velocity, Mass = alldata
            
            # explicitly set an analytic mass-concentration relation
            sats_prof_model = em_models.NFWPhaseSpace(conc_mass_model='dutton_maccio14')
            
            # build the full hod model, and set our params
            base_model = em_models.PrebuiltHodModelFactory('zheng07')
            self.model = em_models.HodModelFactory(baseline_model_instance=base_model, 
                                                   satellites_profile=sats_prof_model)
        
            for param in self.model.param_dict:
                self.model.param_dict[param] = getattr(self, param)
                
            # set the random seed
            if self.seed is not None:
                model_with_random_seed(self.model, self.seed)
        
            # compute the rsd factor from cosmology
            H = self.cosmo.engine.H(self.redshift) / self.cosmo.engine.h
            a = 1./(1+self.redshift)
            self.rsd_factor = 1./(a*H)  # (velocity in km/s) * rsd_factor = redshift space offset in Mpc/h
        
            # FOFGroups velocity is normalized by aH, so un-normalize into km/s
            ### FIXME
            if self.halocat.plugin_name == 'FOFGroups':
                Velocity /= self.rsd_factor
        
            # make the halotools catalog 
            cols                  = {}
            cols['redshift']      = self.redshift
            cols['Lbox']          = self.halocat.BoxSize.max()
            cols['particle_mass'] = getattr(self.halocat, 'm0', 1.0)
            cols['halo_x']        = Position[:,0]
            cols['halo_y']        = Position[:,1]
            cols['halo_z']        = Position[:,2]
            cols['halo_vx']       = Velocity[:,0]
            cols['halo_vy']       = Velocity[:,1]
            cols['halo_vz']       = Velocity[:,2]
            cols['halo_mvir']     = Mass
            cols['halo_rvir']     = sats_prof_model.halo_mass_to_halo_radius(Mass)
            cols['halo_id']       = numpy.arange(len(Position))
            cols['halo_upid']     = numpy.zeros_like(Mass) - 1
            cols['simname']       = self.plugin_name
            self._halotools_cat   = sim_manager.UserSuppliedHaloCatalog(**cols)
            
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "populate an input halo catalog with galaxies using the Zheng et al. 2007 HOD"
        
        s.add_argument("halocat", type=DataSource.from_config,
            help="DataSource representing the `halo` catalog")
        s.add_argument('redshift', type=float,
            help='the redshift of the ')
        s.add_argument("logMmin", type=float, 
            help="minimum mass required for a halo to host a central galaxy")
        s.add_argument("sigma_logM", type=float, 
            help="rate of transition from <Ncen>=0 --> <Ncen>=1")
        s.add_argument("alpha", type=float, 
            help="power law slope of the relation between halo mass and <Nsat>")
        s.add_argument("logM0", type=float, 
            help="Low-mass cutoff in <Nsat>")
        s.add_argument("logM1", type=float, 
            help="characteristic halo mass where <Nsat> begins to assume a power law form")
        s.add_argument("rsd", type=str, choices="xyz", 
            help="the direction to do the redshift distortion")
        s.add_argument("seed", type=int,
            help='the number used to seed the random number generator')
        
    def log_populated_stats(self):
        """
        Log statistics of the populated catalog
        """
        data = self.model.mock.galaxy_table
        if not len(data):
            raise ValueError("cannot log statistics of an empty galaxy catalog")
            
        self.logger.info("populated %d galaxies into halo catalog" %len(data))
        fsat = 1.*(data['gal_type'] == 'satellites').sum()/len(data)
        self.logger.info("  satellite fraction: %.2f" %fsat)
        
        logmass = numpy.log10(data['halo_mvir'])
        self.logger.info("  mean log10 halo mass: %.2f" %logmass.mean())
        self.logger.info("  std log10 halo mass: %.2f" %logmass.std())
    
    def update_and_populate(self, **params):
        """
        Update the Hod parameters and populate a mock
        catalog using the new parameters
        
        This is only done on root 0, which is the only rank
        to have the Hod model
        """
        if self.comm.rank == 0:
            
            # update Hod model
            for name in params:
                if name not in self.model.param_dict:
                    valid = list(self.model.param_dict.keys())
                    raise ValueError("'%s' is not a valid Hod parameter name; valid are: %s" %(name, str(valid)))
                self.model.param_dict[name] = params[name]
            
            if not len(self.model.mock.galaxy_table):
                raise ValueError("populated 0 galaxies into the specified halo catalog -- cannot proceed")
                
            # re-populate
            self.populate_mock()
        
    def populate_mock(self):
        """
        Populate the halo catalog with a new set of galaxies. 
        Each call to this function creates a new galaxy catalog,
        overwriting any existing catalog
        
        This is only done on root 0, which is the only rank
        to have the Hod model
        """  
        if self.comm.rank == 0:
            
            # populate for first time (removing only mass=0)
            if not hasattr(self.model, 'mock'):
                self.model.populate_mock(halocat=self._halotools_cat, Num_ptcl_requirement=1)
            # repopulate
            else:
                self.model.mock.populate()
                
            if not len(self.model.mock.galaxy_table):
                raise ValueError("populated 0 galaxies into the specified halo catalog -- cannot proceed")
            self.log_populated_stats()
    
    def parallel_read(self, columns, full=False):
        """
        Return the positions of galaxies, populated into halos using an Hod
        """
        from nbodykit.distributedarray import ScatterArray
        
        # rank 0 does the work, then scatters
        if self.comm.rank == 0:
            
            # populate the model, if need be
            if not hasattr(self.model, 'mock'):
                self.populate_mock()
        
            # the data
            data = self.model.mock.galaxy_table   
        
            # check valid columns
            valid = ['Position', 'Velocity']
            if any(c not in valid and c not in data.colnames for c in columns):
                args = (self.plugin_name, str(valid + data.colnames))
                raise DataSource.MissingColumn('valid column names for %s are: %s' %args)
        
            # position/velocity
            pos = numpy.vstack([data[c] for c in ['x', 'y', 'z']]).T
            vel = numpy.vstack([data[c] for c in ['vx', 'vy', 'vz']]).T
        
            # transform to redshift space
            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                pos[:,dir] += self.rsd_factor * vel[:,dir] 
        
            toret = []
            for c in columns:
                if c == 'Position':
                    toret.append(pos)
                elif c == 'Velocity':
                    toret.append(vel)
                else:
                    toret.append(numpy.array(data[c].copy()))
        else:
            toret = [None for c in columns]
        
        # scatter the data from rank 0
        newdata = []
        for d in toret:
            newdata.append(ScatterArray(d, self.comm, root=0))

        yield newdata
        
        

            

        
    
