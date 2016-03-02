from nbodykit.extensionpoints import DataSource, datasources
from nbodykit.distributedarray import ScatterArray

import numpy
import logging

logger = logging.getLogger('Zheng07Hod')

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
                    alpha=0.76, logM0=13.27, logM1=14.08, rsd=None):
        """
        Default values for Hod values from Reid et al. 2014
        """
        # load halotools
        try:
            from halotools import sim_manager
        except:
            name = self.__class__.__name__
            raise ValueError("`halotools` must be installed to use '%s' DataSource" %name)
            
        # need cosmology
        if self.cosmo is None:
            raise AttributeError("a cosmology instance is required to populate an Hod")
        
        # set defaults and load the rest
        sim_manager.sim_defaults.default_cosmology = self.cosmo.engine
        sim_manager.sim_defaults.default_redshift = self.redshift
        from halotools import empirical_models as em_models

        # grab the halocat BoxSize
        self.BoxSize = self.halocat.BoxSize
        
        # read data from halo catalog and then gather to root
        Columns = ['Position','Velocity', 'Mass']
        [data] = self.halocat.read(Columns, full=True)
        alldata = [self.comm.gather(d) for d in data]
        
        # rank 0 does the populating
        if self.comm.rank == 0:
            
            alldata = [numpy.concatenate(d, axis=0) for d in alldata]
            Position, Velocity, Mass = alldata
            
            # explicitly set an analytic mass-concentration relation
            sats_prof_model = em_models.NFWPhaseSpace(conc_mass_model='dutton_maccio14')
        
            # build the model and set defaults
            base_model = em_models.PrebuiltHodModelFactory('zheng07')
            self.model = em_models.HodModelFactory(baseline_model_instance=base_model, 
                                                    satellites_profile=sats_prof_model)
            for param in self.model.param_dict:
                self.model.param_dict[param] = getattr(self, param)
        
            # Velocity is normalized by aH, so un-normalize into km/s
            H = self.cosmo.engine.H(self.redshift) / self.cosmo.engine.h
            a = 1./(1+self.redshift)
            self.rsd_factor = 1./(a*H)  # (velocity in km/s) * rsd_factor = redshift space offset in Mpc/h
            Velocity /= self.rsd_factor
        
            # make the halotools catalog 
            cols                  = {}
            cols['redshift']      = self.redshift
            cols['Lbox']          = self.halocat.BoxSize.max()
            cols['particle_mass'] = self.halocat.m0
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
            self._halotools_cat   = sim_manager.UserSuppliedHaloCatalog(**cols)
            
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "DataSource for galaxies populating a halo catalog using the Zheng et al. 2007 Hod"
        
        s.add_argument("halocat", type=datasources.FOFGroups.from_config,
            help="`FOFGroups` DataSource representing the `halo` catalog")
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
            help="the direction to do redshift distortion")
        

    def update_and_populate(self, **params):
        """
        Update the Hod parameters and populate a mock
        catalog using the new parameters
        
        This is only done on root 0, which is the only rank
        to have the Hod model
        """
        if self.rank == 0:
            
            # update Hod model
            for name in params:
                if name not in self.model.param_dict:
                    valid = list(self.model.param_dict.keys())
                    raise ValueError("'%s' is not a valid Hod parameter name; valid are: %s" %(name, str(valid)))
                self.model.param_dict[name] = params[name]
            
            # repopulate
            self.populate_mock()
            N = len(self.model.mock.galaxy_table)
            logger.info("populated %d galaxies into halo catalog" %N)
     
    def populate_mock(self):
        """
        Populate the halo catalog with a new set of galaxies. 
        Each call to this function creates a new galaxy catalog,
        overwriting any existing catalog
        
        This is only done on root 0, which is the only rank
        to have the Hod model
        """  
        if self.rank == 0:
            
            # populate for first time  
            if not hasattr(self.model, 'mock'):
                self.model.populate_mock(halocat=self._halotools_cat)
            # repopulate
            else:
                self.model.mock.populate()
            
            N = len(self.model.mock.galaxy_table)
            logger.info("populated %d galaxies into halo catalog" %N)
    
    def read(self, columns, full=False):
        """
        Return the positions of galaxies, populated into halos using an Hod
        """
        # rank 0 does the work, then scatters
        if self.comm.rank == 0:
            
            # populate the model, if need be
            if not hasattr(self.model, 'mock'):
                self.model.populate_mock(halocat=self._halotools_cat)
                N = len(self.model.mock.galaxy_table)
                logger.info("populated %d galaxies into halo catalog" %N)
        
            # the data
            data = self.model.mock.galaxy_table        
        
            # check valid columns
            valid = ['Position', 'Velocity']
            if any(c not in valid and c not in data.colnames for c in columns):
                args = (self.__class__.__name__, str(valid + data.colnames))
                raise ValueError('valid column names for %s are: %s' %args)
        
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
                    toret.append(numpy.array(data[c]))
        else:
            toret = [None for c in columns]
        
        # scatter the data from rank 0
        newdata = []
        for d in toret:
            newdata.append(ScatterArray(d, self.comm, root=0))
            
        yield newdata
        
        

            

        
    
