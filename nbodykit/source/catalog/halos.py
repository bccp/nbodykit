from .array import ArrayCatalog
from nbodykit import CurrentMPIComm, transform
from nbodykit.utils import GatherArray, ScatterArray
from nbodykit.base.catalog import CatalogSourceBase, CatalogSource, column

import numpy
import logging

class HaloCatalog(CatalogSource):
    """
    A CatalogSource of objects that represent halos, which can be populated
    using analytic models from :mod:`halotools`.

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
    logger = logging.getLogger("HaloCatalog")

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

        # names of the mass and radius fields, based on mass def
        self.attrs['halo_mass_key'] = 'halo_m' + mdef
        self.attrs['halo_radius_key'] = 'halo_r' + mdef

        # the size
        self._size = self._source.size

        # init the base class
        CatalogSource.__init__(self, comm=comm)

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

    def to_halotools(self, BoxSize=None):
        """
        Return the HaloCatalog as a
        :class:`halotools.sim_manager.UserSuppliedHaloCatalog`.

        The Halotools catalog only holds the local data, although halos are
        labeled via the ``halo_id`` column using the global index.

        Parameters
        ----------
        BoxSize : float, array_like, optional
            the size of the box; must be supplied if 'BoxSize' is not in
            the :attr:`attrs` dict

        Returns
        -------
        cat : :class:`halotools.sim_manager.UserSuppliedHaloCatalog`
            the Halotools halo catalog, storing the local halo data
        """
        from halotools.sim_manager import UserSuppliedHaloCatalog

        # make sure we have at least one halo
        if self.csize == 0:
            raise ValueError("cannot create halotools UserSuppliedHaloCatalog; catalog is empty")

        # make sure we have a BoxSize
        if BoxSize is None:
            BoxSize = self.attrs.get('BoxSize', None)
        if BoxSize is None:
            raise ValueError("please specify a 'BoxSize' to convert to a halotools catalog")

        # compute the columns
        cols = ['Position', 'Velocity', 'Mass', 'Radius', 'Concentration']
        cols = self.compute(*[self[col] for col in cols])
        Position, Velocity, Mass, Radius, Concen = [col for col in cols]

        # halo mass and radius keys
        mkey = self.attrs['halo_mass_key']
        rkey = self.attrs['halo_radius_key']

        # global halo ids (across all ranks)
        halo_id = self.Index.compute()

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
        kws['halo_hostid']    = halo_id
        kws['halo_upid']     = numpy.zeros(len(Position)) - 1
        kws['halo_local_id'] = numpy.arange(0, self.size, dtype='i8')

        # add metadata too
        kws['cosmology']     = self.cosmo
        kws['redshift']      = self.attrs['redshift']
        kws['Lbox']          = BoxSize
        kws['particle_mass'] = self.attrs.get('particle_mass', 1.0)
        kws['mdef']          = self.attrs['mdef']

        return UserSuppliedHaloCatalog(**kws)

    def populate(self, model, BoxSize=None, seed=None, **params):
        """
        Populate the HaloCatalog using a :mod:`halotools` model.

        The model can be a built-in model from :mod:`nbodykit.hod` (which
        will be converted to a Halotools model) or directly a Halotools model
        instance.

        This assumes that this is the first time this catalog has been
        populated with the input model. To re-populate using the same
        model (but different parameters), call the :func:`repopulate`
        function of the returned :class:`PopulatedHaloCatalog`.

        Parameters
        ----------
        model : :class:`nbodykit.hod.HODModel` or halotools model object
            the model instance to use to populate; model types from
            :mod:`nbodykit.hod` will automatically be converted
        BoxSize : float, 3-vector, optional
            the box size of the catalog; this must be supplied if 'BoxSize'
            is not in :attr:`attrs`
        seed : int, optional
            the random seed to use when populating the mock
        **params :
            key/value pairs specifying the model parameters to use

        Returns
        -------
        cat : :class:`PopulatedHaloCatalog`
            the catalog object storing information about the populated objects

        Examples
        --------
        Initialize a demo halo catalog:

        >>> from nbodykit.tutorials import DemoHaloCatalog
        >>> cat = DemoHaloCatalog('bolshoi', 'rockstar', 0.5)

        Populate with the built-in Zheng07 model:

        >>> from nbodykit.hod import Zheng07Model
        >>> galcat = cat.populate(Zheng07Model, seed=42)

        And then re-populate galaxy catalog with new parameters:

        >>> galcat.repopulate(alpha=0.9, logMmin=13.5, seed=42)
        """
        from nbodykit.hod import HODModel
        from halotools.empirical_models import ModelFactory
        from halotools.sim_manager import UserSuppliedHaloCatalog

        # handle builtin model types
        if isinstance(model, (type, HODModel)) and issubclass(model, HODModel):
            model = model.to_halotools(self.cosmo, self.attrs['redshift'],
                                        self.attrs['mdef'], concentration_key='halo_nfw_conc')

        # check model type
        if not isinstance(model, ModelFactory):
            raise TypeError("model for populating mocks should be a Halotools ModelFactory")

        # make halotools catalog
        halocat = self.to_halotools(BoxSize=BoxSize)

        # gather the halo data to root
        all_halos = GatherArray(halocat.halo_table.as_array(), self.comm, root=0)

        # only the root rank needs to store the halo data
        if self.comm.rank == 0:
            data = {col:all_halos[col] for col in all_halos.dtype.names}
            data.update({col:getattr(halocat, col) for col in ['Lbox', 'redshift', 'particle_mass']})
            halocat = UserSuppliedHaloCatalog(**data)
        else:
            halocat = None

        # cache the model so we have option to call repopulate later
        self.model = model

        # return the populated catalog
        return _populate_mock(self, model, seed=seed, halocat=halocat, **params)


class PopulatedHaloCatalog(ArrayCatalog):
    """
    A CatalogSource to represent a set of objects populated into a
    :class:`HaloCatalog`.

    .. note::
        Users should not access this class directly, but rather, call
        :func:`HaloCatalog.populate` to generate a :class:`PopulatedHaloCatalog`.

    Parameters
    ----------
    data : structured numpy.ndarray
        the data of the populated objects
    model :
        the Halotools model instance
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology`
        the cosmology instance
    """
    @CurrentMPIComm.enable
    def __init__(self, data, model, cosmo, comm=None):
        ArrayCatalog.__init__(self, data, comm=comm)
        self.model = model
        self.cosmo = cosmo

    def repopulate(self, seed=None, **params):
        """
        Re-populate the catalog in-place, using the specified ``seed``
        or model parameters.

        This re-uses the model that was last used to create this catalog.
        It is faster than :func:`HaloCatalog.populate` as it avoids
        initialization steps. It is intended to be used when looping over
        different parameter sets, e.g., when performing parameter optimization.

        .. note::
            This operation is performed in-place.

        Parameters
        ----------
        seed : int, optional
            the random seed to use when populating the mock
        **params :
            key/value pairs specifying the model parameters to use
        """
        _populate_mock(self, self.model, seed=seed, inplace=True, **params)


def _populate_mock(cat, model, seed=None, halocat=None, inplace=False, **params):
    """
    Internal function to perform the mock population on a HaloCatalog, given
    a :mod:`halotools` model.

    The implementation is not massively parallel. The data is gathered to
    the root rank, mock population is performed, and then the data is
    re-scattered evenly across ranks.
    """
    # verify input params
    valid = sorted(model.param_dict)
    missing = set(params) - set(valid)
    if len(missing):
        raise ValueError("invalid halo model parameter names: %s" % str(missing))

    # update the model parameters
    model.param_dict.update(params)

    # set the seed randomly if it is None
    if seed is None:
        if cat.comm.rank == 0:
            seed = numpy.random.randint(0, 4294967295)
        seed = cat.comm.bcast(seed)

    # the types of galaxies we are populating
    gal_types = getattr(model, 'gal_types', [])

    exception = None
    try:
        # the root will do the mock population
        if cat.comm.rank == 0:

            # re-populate the mock (without halo catalog pre-processing)
            kws = {'seed':seed, 'Num_ptcl_requirement':0, 'halo_mass_column_key':cat.attrs['halo_mass_key']}
            if hasattr(model, 'mock'):
                model.mock.populate(**kws)
            # populating model for the first time (initialization costs)
            else:
                if halocat is None:
                    raise ValueError("halocat cannot be None if we are populating for the first time")
                model.populate_mock(halocat=halocat, **kws)

            # enumerate gal types as integers
            # NOTE: necessary to avoid "O" type columns
            _enum_gal_types(model.mock.galaxy_table, gal_types)

            # crash if any object dtypes
            # NOTE: we cannot use GatherArray/ScatterArray on objects
            data = _test_for_objects(model.mock.galaxy_table).as_array()

        else:
            data = None

    except Exception as e:
        exception = e

    # re-raise the error
    exception = cat.comm.bcast(exception, root=0)
    if exception is not None:
        raise exception

    # re-scatter the data evenly
    data = ScatterArray(data, cat.comm, root=0)

    # re-initialize with new source
    if inplace:
        PopulatedHaloCatalog.__init__(cat, data, model, cat.cosmo, comm=cat.comm)
        galcat = cat
    else:
        galcat = PopulatedHaloCatalog(data, model, cat.cosmo, comm=cat.comm)

    # crash with no particles!
    if galcat.csize == 0:
        raise ValueError("no particles in catalog after populating halo catalog")

    # add Position, Velocity
    galcat['Position'] = transform.StackColumns(galcat['x'], galcat['y'], galcat['z'])
    galcat['Velocity'] = transform.StackColumns(galcat['vx'], galcat['vy'], galcat['vz'])

    # add VelocityOffset
    z = cat.attrs['redshift']
    rsd_factor = (1+z) / (100.*cat.cosmo.efunc(z))
    galcat['VelocityOffset'] = galcat['Velocity'] * rsd_factor

    # add meta-data
    galcat.attrs.update(cat.attrs)
    galcat.attrs.update(model.param_dict)
    galcat.attrs['seed'] = seed
    galcat.attrs['gal_types'] = {t:i for i,t in enumerate(gal_types)}

    # propagate total number of halos for logging
    if galcat.comm.rank == 0:
        Nhalos = len(galcat.model.mock.halo_table)
    else:
        Nhalos = None
    Nhalos = galcat.comm.bcast(Nhalos, root=0)

    # and log some info
    _log_populated_stats(galcat, Nhalos)

    return galcat

def _log_populated_stats(cat, Nhalos):
    """
    Internal function to log statistics of a populated catalog. It logs
    information about satellite fraction, mass distribution, and total
    number.

    This is called each time that :func:`_populate_mock` is called.
    """
    # compute the satellite fraction
    fsat = None
    if 'gal_type' in cat:
        gal_types = cat.attrs['gal_types']
        if 'satellites' in gal_types:
            Nsats = cat.comm.allreduce((cat['gal_type'] == gal_types['satellites']).sum())
            fsat = float(Nsats) / cat.csize
            cat.attrs['fsat'] = fsat

    # mass distribution stats
    mass = cat[cat.attrs['halo_mass_key']].compute()
    logmass = numpy.log10(mass)
    avg_logmass = cat.comm.allreduce(logmass.sum()) / cat.csize
    sq_logmass = cat.comm.allreduce(((logmass - avg_logmass)**2).sum()) / cat.csize
    std_logmass = sq_logmass**0.5

    if cat.comm.rank == 0:
        if fsat is not None:
            cat.logger.info("satellite fraction: %.2f" % fsat)
        cat.logger.info("populated %d objects into %d halos" % (cat.csize, Nhalos))
        cat.logger.info("mean log10 halo mass: %.2f" % avg_logmass)
        cat.logger.info("std log10 halo mass: %.2f" % std_logmass)

def _enum_gal_types(galtab, types):
    """
    Enumerate the galaxy types as integers instead of strings.
    """
    if 'gal_type' in galtab.colnames:
        gal_type = numpy.zeros(len(galtab), dtype='i4')
        for i, gtype in enumerate(sorted(types[1:])):
            idx = galtab['gal_type'] == gtype
            gal_type[idx] = i+1
        galtab.replace_column('gal_type', gal_type)

def _test_for_objects(data):
    """
    Raise an exception if any of the columns have 'O' dtype.

    This is necessary because the 'O' dtype has an undefined size and cannot
    be scattered/gathered via MPI.
    """
    for col in data.colnames:
        if data.dtype[col] == 'O':
            raise TypeError("column '%s' is of type 'O'; must convert to integer or string" %col)
    return data
