from .halos import HalotoolsMockCatalog
from nbodykit.base.catalog import column
from nbodykit import CurrentMPIComm
from nbodykit.transform import StackColumns

import abc
import logging
import numpy

class HODCatalogBase(HalotoolsMockCatalog):
    """
    A base class to be used for HOD population of a halo catalog.

    The user must supply the :func:`get_model` function, which returns
    the halotools composite HOD model.

    This abstraction allows the user to implement several
    different types of HOD models, while using the population
    framework of this base class.
    """
    logger = logging.getLogger("HODCatalogBase")

    @CurrentMPIComm.enable
    def __init__(self, halos, seed=None, use_cache=False, comm=None, **params):

        from astropy.cosmology import FLRW
        from halotools.empirical_models import model_defaults
        from halotools.sim_manager import UserSuppliedHaloCatalog, CachedHaloCatalog

        # check input type
        if not isinstance(halos, (UserSuppliedHaloCatalog, CachedHaloCatalog)):
            raise TypeError(("input halos catalog for HalotoolsMockCatalog should be "
                             "halotools UserSuppliedHaloCatalog or CachedHaloCatalog"))

        # try to extract meta-data from catalog
        mdef     = getattr(halos, 'mdef', 'vir')
        cosmo    = getattr(halos, 'cosmology', None)
        Lbox     = getattr(halos, 'Lbox', None)
        redshift = getattr(halos, 'redshift', None)

        # fail if we are missing any of these
        required = ['cosmology', 'Lbox', 'redshift']
        for name, attr in zip(required, [cosmo, Lbox, redshift]):
            if attr is None:
                raise AttributeError("input UserSuppliedHaloCatalog must have '%s attribute" %name)

        # promote astropy cosmology to nbodykit's Cosmology
        if isinstance(cosmo, FLRW):
            from nbodykit.cosmology import Cosmology
            cosmo = Cosmology.from_astropy(cosmo)
        self.cosmo = cosmo

        # store the halotools catalog
        self.halos = halos

        # store mass and radius keys
        self.mass   = model_defaults.get_halo_mass_key(mdef)
        self.radius = model_defaults.get_halo_boundary_key(mdef)

        # check for any missing columns
        needed = ['halo_%s' %s for s in ['x','y','z','vx','vy','vz', 'id', 'upid']]
        needed += [self.mass, self.radius]
        missing = set(needed) - set(halos.halo_table.colnames)
        if len(missing):
            raise ValueError("missing columns from halotools UserSuppliedHaloCatalog: %s" %str(missing))

        # store the attributes
        self.attrs['BoxSize'] = Lbox
        self.attrs['mdef'] = mdef
        self.attrs['redshift'] = redshift
        self.attrs['cosmo'] = dict(cosmo)

        # make the actual source
        HalotoolsMockCatalog.__init__(self, halos, self.get_model(), seed=seed,
                                        comm=comm, use_cache=use_cache, **params)

    @abc.abstractmethod
    def get_model(self):
        """
        Abstract function to be overwritten by user; this should return
        the HOD model instance that will be used to do the mock
        population.

        See :ref:`the documentation <custom-hod-mocks>` for more details.

        Returns
        -------
        :class:`~halotools.empirical_models.HodModelFactory`
            the halotools object implementing the HOD model
        """
        pass

    def repopulate(self, seed=None, **params):
        """
        Update the model parameters and then re-populate the mock catalog.

        .. warning::
            This operation is done in-place, so the size of the
            CatalogSource changes.

        Parameters
        ----------
        seed : int, optional
            the new seed to use when populating the mock
        Num_ptcl_requirement : int, optional
        **params :
            key/value pairs of HOD parameters to update
        """
        # verify input params
        valid = sorted(self.model.param_dict)
        missing = set(params) - set(valid) 
        if len(missing):
            raise ValueError("invalid HOD parameter names: %s" % str(missing) )

        # re-populate
        HalotoolsMockCatalog.repopulate(self, seed=seed, Num_ptcl_requirement=0,
                                        halo_mass_column_key=self.mass, **params)

        # we dont need the galaxy_table copy
        # data is stored as "_source" attribute internally
        if self.comm.rank == 0:
            del self.model.mock.galaxy_table

        # and log
        self.log_populated_stats()

    def log_populated_stats(self):
        """
        Log statistics of the populated catalog. This is called each
        time that :func:`repopulate` is called.

        Users can override this function in subclasses for custom logging.
        """
        # compute the satellite fraction
        Nsats = self.comm.allreduce((self['gal_type'] == 1).sum())
        fsat = float(Nsats) / self.csize
        self.attrs['fsat'] = fsat

        # mass distribution stats
        mass = self[self.mass].compute()
        logmass = numpy.log10(mass)
        avg_logmass = self.comm.allreduce(logmass.sum()) / self.csize
        sq_logmass = self.comm.allreduce(((logmass - avg_logmass)**2).sum()) / self.csize
        std_logmass = sq_logmass**0.5

        if self.comm.rank == 0:
            self.logger.info("populated %d objects into %d halos" % (self.csize, len(self.model.mock.halo_table)))
            self.logger.info("satellite fraction: %.2f" % fsat)
            self.logger.info("mean log10 halo mass: %.2f" % avg_logmass)
            self.logger.info("std log10 halo mass: %.2f" % std_logmass)

    @column
    def VelocityOffset(self):
        """
        The RSD velocity offset, in units of Mpc/h.

        This multiplies Velocity by 1 / (a*100*E(z)) = 1 / (a H(z)/h)
        """
        z = self.attrs['redshift']
        rsd_factor = (1+z) / (100*self.cosmo.efunc(z))
        return self['Velocity'] * rsd_factor

class HODCatalog(HODCatalogBase):
    """
    A CatalogSource that uses :mod:`halotools` and the HOD prescription of
    Zheng et al 2007 to populate an input halo catalog with galaxies.

    The mock population is done using :mod:`halotools`. See the documentation
    for :class:`halotools.empirical_models.Zheng07Cens` and
    :class:`halotools.empirical_models.Zheng07Sats` for further details
    regarding the HOD.

    The columns generated in this catalog are:

    #. **Position**: the galaxy position
    #. **Velocity**: the galaxy velocity
    #. **VelocityOffset**: the RSD velocity offset, in units of distance
    #. **conc_NFWmodel**: the concentration of the halo
    #. **gal_type**: the galaxy type, 0 for centrals and 1 for satellites
    #. **halo_id**: the global ID of the halo this galaxy belongs to, between 0 and :attr:`csize`
    #. **halo_local_id**: the local ID of the halo this galaxy belongs to, between 0 and :attr:`size`
    #. **halo_mvir**: the halo mass
    #. **halo_nfw_conc**: alias of ``conc_NFWmodel``
    #. **halo_num_centrals**: the number of centrals that this halo hosts, either 0 or 1
    #. **halo_num_satellites**: the number of satellites that this halo hosts
    #. **halo_rvir**: the halo radius
    #. **halo_upid**: equal to -1; should be ignored by the user
    #. **halo_vx, halo_vy, halo_vz**: the three components of the halo velocity
    #. **halo_x, halo_y, halo_z**: the three components of the halo position
    #. **host_centric_distance**: the distance from this galaxy to the center of the halo
    #. **vx, vy, vz**: the three components of the galaxy velocity, equal to ``Velocity``
    #. **x,y,z**: the three components of the galaxy position, equal to ``Position``

    For futher details, please see the :ref:`documentation <hod-mock-data>`.

    .. note::
         Default HOD values are from
         `Reid et al. 2014 <https://arxiv.org/abs/1404.3742>`_

    Parameters
    ----------
    halos : :class:`~halotools.sim_manager.UserSuppliedHaloCatalog`
        the halotools table holding the halo data; this object must have
        the following attributes: `cosmology`, `Lbox`, `redshift`
    logMmin : float, optional
        Minimum mass required for a halo to host a central galaxy
    sigma_logM : float, optional
        Rate of transition from <Ncen>=0 --> <Ncen>=1
    alpha : float, optional
        Power law slope of the relation between halo mass and <Nsat>
    logM0 : float, optional
        Low-mass cutoff in <Nsat>
    logM1 : float, optional
        Characteristic halo mass where <Nsat> begins to assume a power law form
    seed : int, optional
        the random seed to generate deterministic mocks

    References
    ----------
    `Zheng et al. (2007), arXiv:0703457 <https://arxiv.org/abs/astro-ph/0703457>`_
    """
    logger = logging.getLogger("HODCatalog")

    @CurrentMPIComm.enable
    def __init__(self, halos, logMmin=13.031, sigma_logM=0.38,
                    alpha=0.76, logM0=13.27, logM1=14.08,
                    seed=None, use_cache=False, comm=None):

        params = {}
        params['logMmin'] = logMmin
        params['sigma_logM'] = sigma_logM
        params['alpha'] = alpha
        params['logM0'] = logM0
        params['logM1'] = logM1

        HODCatalogBase.__init__(self, halos, seed=seed, use_cache=use_cache, comm=comm, **params)

    def __repr__(self):
        names = ['logMmin', 'sigma_logM', 'alpha', 'logM0', 'logM1']
        s = ', '.join(['%s=%.2f' %(k,self.attrs[k]) for k in names])
        return "HODCatalog(%s)" %s

    def get_model(self):
        """
        Return the Zheng 07 HOD model.

        This model evaluates Eqs. 2 and 5 of Zheng et al. 2007.
        """
        from halotools.empirical_models import HodModelFactory
        from halotools.empirical_models import Zheng07Sats, Zheng07Cens
        from halotools.empirical_models import NFWPhaseSpace, TrivialPhaseSpace

        model = {}

        # use concentration from halo table
        if 'halo_nfw_conc' in self.halos.halo_table.colnames:
            conc_mass_model = 'direct_from_halo_catalog'
        # use empirical prescription for c(M)
        else:
            conc_mass_model = 'dutton_maccio14'

        # occupation functions
        cenocc = Zheng07Cens(prim_haloprop_key=self.mass)
        satocc = Zheng07Sats(prim_haloprop_key=self.mass, modulate_with_cenocc=True, cenocc_model=cenocc)
        satocc._suppress_repeated_param_warning = True

        # add to model
        model['centrals_occupation'] = cenocc
        model['satellites_occupation'] = satocc

        # profile functions
        kws = {'cosmology':self.cosmo.to_astropy(), 'redshift':self.attrs['redshift'], 'mdef':self.attrs['mdef']}
        model['centrals_profile'] = TrivialPhaseSpace(**kws)
        model['satellites_profile'] = NFWPhaseSpace(conc_mass_model=conc_mass_model, **kws)

        return HodModelFactory(**model)
