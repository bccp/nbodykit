from .file import HDFCatalog
from .array import ArrayCatalog
from nbodykit import CurrentMPIComm, transform
from nbodykit.cosmology import Cosmology
from nbodykit.utils import GatherArray, ScatterArray
from nbodykit.base.catalog import CatalogSourceBase, CatalogSource, column

import numpy
import logging

class HaloCatalog(CatalogSource):
    """
    A wrapper CatalogSource of halo objects to interface nicely with
    :class:`halotools.sim_manager.UserSuppliedHaloCatalog`.

    Parameters
    ----------
    source : CatalogSource
        the source holding the particles to be interpreted as halos
    cosmo : :class:`~nbodykit.cosmology.cosmology.Cosmology`
        the cosmology instance;
    redshift : float
        the redshift of the halo catalog
    mdef : str, optional
        string specifying mass definition, used for computing default
        halo radii and concentration; should be 'vir' or 'XXXc' or
        'XXXm' where 'XXX' is an int specifying the overdensity
    mass : str, optional
        the column name specifying the mass of each halo
    position : str, optional
        the column name specifying the position of each halo
    velocity : str, optional
        the column name specifying the velocity of each halo
    """
    def __init__(self, source, cosmo, redshift, mdef='vir',
                 mass='Mass', position='Position', velocity='Velocity'):

        # make sure all of the columns are there
        required = ['mass', 'position', 'velocity']
        for name, col in zip(required, [mass, position, velocity]):
            if col is None:
                raise ValueError("the %s column cannot be None in HaloCatalog" %name)
            if col not in source:
                raise ValueError("input source is missing the %s column; '%s' does not exist" %(name, col))

        if not isinstance(source, CatalogSourceBase):
            raise TypeError("input source to HalotoolsCatalog should be a CatalogSource")

        comm = source.comm
        self._source = source
        self.cosmo = cosmo

        # get the attrs from the source
        self.attrs.update(source.attrs)

        # and save the parameters
        self.attrs['redshift'] = redshift
        self.attrs['cosmo']    = dict(self.cosmo)
        self.attrs['mass']     = mass
        self.attrs['velocity'] = velocity
        self.attrs['position'] = position
        self.attrs['mdef']     = mdef

        # the size
        self._size = self._source.size

        # init the base class
        CatalogSource.__init__(self, comm=comm, use_cache=False)

    @column
    def Mass(self):
        """
        The halo mass column, assumed to be in units of :math:`M_\odot/h`.
        """
        return self._source[self.attrs['mass']]

    @column
    def Position(self):
        """
        The halo position column, assumed to be in units of :math:`\mathrm{Mpc}/h`.
        """
        return self._source[self.attrs['position']]

    @column
    def Velocity(self):
        """
        The halo velocity column, assumed to be in units of km/s.
        """
        return self._source[self.attrs['velocity']]

    @column
    def VelocityOffset(self):
        """
        The redshift-space distance offset due to the velocity in units of
        distance. The assumed units are :math:`\mathrm{Mpc}/h`.

        This multiplies ``Velocity`` by :math:`1 / (a 100 E(z)) = 1 / (a H(z)/h)`.
        """
        z = self.attrs['redshift']
        rsd_factor = (1+z) / (100*self.cosmo.efunc(z))
        return self['Velocity'] * rsd_factor

    @column
    def Concentration(self):
        """
        The halo concentration, computed using :func:`nbodykit.transform.HaloConcentration`.

        This uses the analytic formulas for concentration from
        `Dutton and Maccio 2014 <https://arxiv.org/abs/1402.7073>`_.

        Users can override this column to implement custom mass-concentration
        relations.
        """
        z = self.attrs['redshift']
        mdef = self.attrs['mdef']
        return transform.HaloConcentration(self['Mass'], self.cosmo, z, mdef=mdef)

    @column
    def Radius(self):
        """
        The halo radius, computed using :func:`nbodykit.transform.HaloRadius`.

        Assumed units of :math:`\mathrm{Mpc}/h`.
        """
        z = self.attrs['redshift']
        mdef = self.attrs['mdef']
        return transform.HaloRadius(self['Mass'], self.cosmo, z, mdef=mdef)

    def to_halotools(self, BoxSize=None, selection='Selection'):
        """
        Return the CatalogSource as a
        :class:`halotools.sim_manager.UserSuppliedHaloCatalog`.

        The Halotools catalog only holds the local data, although halos are
        labeled via the ``halo_id`` column using the global index.

        Parameters
        ----------
        BoxSize : float, array_like, optional
            the size of the box; note that anisotropic boxes are currently
            not supported by halotools
        selection : str, optional
            the name of the column to slice the data on before converting
            to a halotools catalog

        Returns
        -------
        cat : :class:`halotools.sim_manager.UserSuppliedHaloCatalog`
            the Halotools halo catalog, storing the local halo data
        """
        from halotools.sim_manager import UserSuppliedHaloCatalog
        from halotools.empirical_models import model_defaults

        # make sure we have at least one halo
        if self.csize == 0:
            raise ValueError("cannot create halotools UserSuppliedHaloCatalog; catalog is empty")

        # make sure we have a BoxSize
        if BoxSize is None:
            BoxSize = self.attrs.get('BoxSize', None)
        if BoxSize is None:
            raise ValueError("please specify a 'BoxSize' to convert to a halotools catalog")

        # make sure selection exists
        assert selection in self, "'%s' selection column is not valid" %selection

        # slice the catalog
        sel = self[selection]
        cat = self[sel]

        # compute the columns
        cols = ['Position', 'Velocity', 'Mass', 'Radius', 'Concentration']
        cols = cat.compute(*[cat[col][sel] for col in cols])
        Position, Velocity, Mass, Radius, Concen = [col for col in cols]

        # names of the mass and radius fields, based on mass def
        mkey = model_defaults.get_halo_mass_key(self.attrs['mdef'])
        rkey = model_defaults.get_halo_boundary_key(self.attrs['mdef'])

        # global halo ids (across all ranks)
        halo_id = cat.Index.compute()

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
        kws['halo_local_id'] = numpy.arange(0, cat.size, dtype='i8')

        # add metadata too
        kws['cosmology']     = self.cosmo
        kws['redshift']      = self.attrs['redshift']
        kws['Lbox']          = BoxSize
        kws['particle_mass'] = self.attrs.get('particle_mass', 1.0)
        kws['mdef']          = self.attrs['mdef']

        return UserSuppliedHaloCatalog(**kws)

class HalotoolsCachedCatalog(HDFCatalog):
    """
    Load one of the built-in :mod:`halotools` catalogs using
    :class:`~halotools.sim_manager.CachedHaloCatalog`.

    Parameters
    ----------
    simname : string
        Nickname of the simulation. Currently supported simulations are
        Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
        MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).
    halo_finder : string
        Nickname of the halo-finder, e.g. `rockstar` or `bdm`.
    redshift : float
        Redshift of the requested snapshot.
        Must match one of theavailable snapshots within ``dz_tol``f,
        or a prompt will be issued providing the nearest
        available snapshots to choose from.

    Examples
    --------
    >>> cat = HalotoolsCachedCatalog('bolshoi', 'rockstar', 0.5)
    >>> halotools_cat = cat.to_halotools()
    """
    @CurrentMPIComm.enable
    def __init__(self, simname, halo_finder, redshift, comm=None, use_cache=False):

        from halotools.sim_manager import CachedHaloCatalog, DownloadManager
        from halotools.sim_manager.cached_halo_catalog import InvalidCacheLogEntry
        from halotools.sim_manager.supported_sims import supported_sim_dict

        # the comm
        self.comm = comm

        # try to automatically load from the Halotools cache
        exception = None
        if self.comm.rank == 0:
            try:
                cached_halos = CachedHaloCatalog(simname=simname, halo_finder=halo_finder, redshift=redshift)
                fname = cached_halos.fname
            except InvalidCacheLogEntry:

                # try to download on the root rank
                    try:
                        # download
                        dl = DownloadManager()
                        dl.download_processed_halo_table(simname, halo_finder, redshift)

                        # access the cached halo catalog and get fname attribute
                        # NOTE: this does not read the data
                        cached_halos = CachedHaloCatalog(simname=simname, halo_finder=halo_finder, redshift=redshift)
                        fname = cached_halos.fname
                    except Exception as e:
                        exception = e
        else:
            fname = None

        # re-raise a download error on all ranks if it occurred
        exception = self.comm.bcast(exception, root=0)
        if exception is not None:
            raise exception

        # broadcast the file we are loading
        fname = self.comm.bcast(fname, root=0)

        # initialize an HDF catalog
        HDFCatalog.__init__(self, fname, comm=comm, use_cache=use_cache)

        # add some meta-data if it exists
        self.attrs['BoxSize'] = cached_halos.Lbox
        self.attrs['redshift'] = cached_halos.redshift
        self.attrs['particle_mass'] = cached_halos.particle_mass

        # add the cosmology
        cosmo = supported_sim_dict[simname]().cosmology
        self.cosmo = Cosmology.from_astropy(cosmo)
        self.attrs['cosmo'] = dict(self.cosmo)

    @column
    def Position(self):
        """
        Halo positions, in units of Mpc/h.
        """
        pos = transform.StackColumns(self['halo_x'], self['halo_y'], self['halo_z'])
        return self.make_column(pos)

    @column
    def Velocity(self):
        """
        Halo velocity, in units of km/s.
        """
        vel = transform.StackColumns(self['halo_vx'], self['halo_vy'], self['halo_vz'])
        return self.make_column(vel)

    def to_halotools(self):
        """
        Convert the input CatalogSource to a halotools ``UserSuppliedHaloCatalog``.
        """
        from halotools.sim_manager import UserSuppliedHaloCatalog

        # NOTE: include all columns that start with halo_ since halotools requires this
        # this means defaults + Position/Velocity won't be added
        columns = [col for col in self if col.startswith('halo_')]
        data = dict(zip(columns, self.compute(*[self[col] for col in columns])))

        # add the meta-data
        data['Lbox'] = self.attrs['BoxSize']
        data['redshift'] = self.attrs['redshift']
        data['particle_mass'] = self.attrs['particle_mass']
        data['cosmology'] = self.cosmo.to_astropy()

        return UserSuppliedHaloCatalog(**data)


class HalotoolsMockCatalog(ArrayCatalog):
    """
    A catalog of objects generated from a :mod:`halotools` model and halo
    catalog.

    Mock population is not massively parallel; halo data is gathered to the
    root rank, which performs the mock population step, and then the data
    is re-scattered evenly amongst the available ranks.

    The :func:`repopulate` function can be called with new parameters to
    re-populate halos in-place.

    .. note::
        This class does not verify that the input catalog and model are
        compatible. Rather, it simply raises a runtime exception if the mock
        population step fails.

    Parameters
    ----------
    halos : UserSuppliedHaloCatalog or CachedHaloCatalog
        the halotools table holding the halo data
    model :
        the halotools model instance; mock population done via the
        :func:`populate_mock`
    seed : int, optional
        the random seed to generate deterministic mocks
    **params :
        key/value pairs that specify the model parameters

    Examples
    --------
    First, load a cached halotools catalog:

    >>> cat = HalotoolsCachedCatalog('bolshoi', 'rockstar', 0.5)
    >>> halotools_cat = cat.to_halotools() # this is a halotools UserSuppliedHaloCatalog

    Then, initialize a halotools model at the same redshift:

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> zheng07_model = PrebuiltHodModelFactory('zheng07', threshold = -19.5, redshift = 0.5)

    Finally, create the mock catalog:

    >>> mock = HalotoolsMockCatalog(halotools_cat, zheng07_model)
    """
    logger = logging.getLogger("HalotoolsMockCatalog")

    @CurrentMPIComm.enable
    def __init__(self, halos, model, seed=None, use_cache=False, comm=None, **params):

        from halotools.empirical_models import model_defaults
        from halotools.sim_manager import UserSuppliedHaloCatalog, CachedHaloCatalog

        # check input type
        if not isinstance(halos, (UserSuppliedHaloCatalog, CachedHaloCatalog)):
            raise TypeError(("input halos catalog for HalotoolsMockCatalog should be "
                             "halotools UserSuppliedHaloCatalog or CachedHaloCatalog"))

        self.comm = comm
        self.use_cache = use_cache

        # store the model (and delete any existing mocks)
        if hasattr(model, 'mock'): del model.mock
        self.model = model

        # grab the BoxSize from the halotools catalog
        self.attrs['BoxSize'] = halos.Lbox
        self.attrs['redshift'] = halos.redshift

        # shift halo data to the root rank
        from astropy.table import Table

        # gather the halo data
        halo_table = test_for_objects(halos.halo_table)
        all_halos = GatherArray(halo_table.as_array(), self.comm, root=0)

        # only the root rank needs to store the halo data
        if self.comm.rank == 0:
            data = {col:all_halos[col] for col in all_halos.dtype.names}
            data.update({col:getattr(halos, col) for col in ['Lbox', 'redshift', 'particle_mass']})
            self.halos = UserSuppliedHaloCatalog(**data)
        else:
            self.halos = None

        # populate the mock and init the base class
        self.repopulate(seed=seed, **params)

    def repopulate(self, seed=None, **kwargs):
        """
        Update the model parameters and then re-populate the mock catalog.

        .. warning::
            This operation is done in-place, so the size of the
            CatalogSource changes.

        Parameters
        ----------
        seed : int, optional
            the new seed to use when populating the mock
        **kwargs :
            key/value pairs of HOD parameters to update; any keywords that
            are not valid parameters are passed to the :func:`populate_mock`
            function.
        """
        # set the seed randomly if it is None
        if seed is None:
            if self.comm.rank == 0:
                seed = numpy.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs['seed'] = seed

        # update the model parameters
        params = {k:kwargs.pop(k) for k in list(kwargs) if k in self.model.param_dict}
        self.model.param_dict.update(params)
        self.attrs.update(params)

        exception = None
        try:
            # the root will do the mock population
            if self.comm.rank == 0:

                # re-populate the mock (without halo catalog pre-processing)
                if hasattr(self.model, 'mock'):
                    self.model.mock.populate(seed=self.attrs['seed'], **kwargs)
                else:
                    self.model.populate_mock(halocat=self.halos, seed=self.attrs['seed'], **kwargs)
                    del self.halos.halo_table

                # enumerate gal types as integers
                enum_gal_types(self.model.mock.galaxy_table, getattr(self.model, 'gal_types', []))

                # crash if any object dtypes
                data = test_for_objects(self.model.mock.galaxy_table).as_array()
            else:
                data = None

        except Exception as e:
            exception = e

        # re-raise the error
        exception = self.comm.bcast(exception, root=0)
        if exception is not None:
            raise exception

        # re-scatter the data evenly
        data = ScatterArray(data, self.comm, root=0)

        # re-initialize with new source
        ArrayCatalog.__init__(self, data, comm=self.comm, use_cache=self.use_cache)

        # crash with no particles!
        if self.csize == 0:
            raise ValueError("no particles in catalog after populating halo catalog")

    @column
    def Position(self):
        """
        Galaxy positions, in units of Mpc/h.
        """
        pos = transform.StackColumns(self['x'], self['y'], self['z'])
        return self.make_column(pos)

    @column
    def Velocity(self):
        """
        Galaxy velocity, in units of km/s.
        """
        vel = transform.StackColumns(self['vx'], self['vy'], self['vz'])
        return self.make_column(vel)

def enum_gal_types(galtab, types):
    """
    Enumerate the galaxy types as integers instead of strings.
    """
    if 'gal_type' in galtab.colnames:
        gal_type = numpy.zeros(len(galtab), dtype='i4')
        for i, gtype in enumerate(sorted(types[1:])):
            idx = galtab['gal_type'] == gtype
            gal_type[idx] = i+1
        galtab.replace_column('gal_type', gal_type)

def test_for_objects(data):
    """
    Raise an exception if any of the columns have 'O' dtype.

    This is necessary because the 'O' dtype has an undefined size and cannot
    be scattered/gathered via MPI.
    """
    for col in data.colnames:
        if data.dtype[col] == 'O':
            raise TypeError("column '%s' is of type 'O'; must convert to integer or string" %col)
    return data
