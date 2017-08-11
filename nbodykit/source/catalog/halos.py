from nbodykit.base.catalog import CatalogSource, column
from nbodykit import transform
import numpy

class HaloCatalog(CatalogSource):
    """
    A wrapper CatalogSource of halo objects to interface nicely with
    :class:`halotools.sim_manager.UserSuppliedHaloCatalog`.

    Parameters
    ----------
    source : CatalogSource
        the source holding the particles to be interpreted as halos
    cosmo : :class:`~nbodykit.cosmology.core.Cosmology`
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
        for name, col in zip(['mass', 'position', 'velocity'], [mass, position, velocity]):
            if col is None:
                raise ValueError("the %s column cannot be None in HaloCatalog" %name)
            if col not in source:
                raise ValueError("input source is missing the %s column; '%s' does not exist" %(name, col))

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
        CatalogSource.__init__(self, comm=comm, use_cache=False)

    @property
    def size(self):
        return self._source.size

    @column
    def Mass(self):
        """
        The halo mass column, assumed to be in units of :math:`M_\odot/h`.
        """
        return self.make_column(self._source[self.attrs['mass']])

    @column
    def Position(self):
        """
        The halo position column, assumed to be in units of :math:`\mathrm{Mpc}/h`.
        """
        return self.make_column(self._source[self.attrs['position']])

    @column
    def Velocity(self):
        """
        The halo velocity column, assumed to be in units of km/s.
        """
        return self.make_column(self._source[self.attrs['velocity']])

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
        Return the CatalogSource as a :class:`halotools.sim_manager.UserSuppliedHaloCatalog`.

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
