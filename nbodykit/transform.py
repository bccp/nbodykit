import numpy
import dask.array as da
from six import string_types
from nbodykit.utils import deprecate
from nbodykit import _global_options
def StackColumns(*cols):
    """
    Stack the input dask arrays vertically, column by column.

    This uses :func:`dask.array.vstack`.

    Parameters
    ----------
    *cols : :class:`dask.array.Array`
        the dask arrays to stack vertically together

    Returns
    -------
    :class:`dask.array.Array` :
        the dask array where columns correspond to the input arrays

    Raises
    ------
    TypeError
        If the input columns are not dask arrays
    """
    cols = da.broadcast_arrays(*cols)
    return da.vstack(cols).T

def ConcatenateSources(*sources, **kwargs):
    """
    Concatenate CatalogSource objects together, optionally including only
    certain columns in the returned source.

    .. note::
        The returned catalog object carries the meta-data from only
        the first catalog supplied to this function (in the ``attrs`` dict).

    Parameters
    ----------
    *sources : subclass of :class:`~nbodykit.base.catalog.CatalogSource`
        the catalog source objects to concatenate together
    columns : str, list of str, optional
        the columns to include in the concatenated catalog

    Returns
    -------
    CatalogSource :
        the concatenated catalog source object

    Examples
    --------
    >>> from nbodykit.lab import *
    >>> source1 = UniformCatalog(nbar=100, BoxSize=1.0)
    >>> source2 = UniformCatalog(nbar=100, BoxSize=1.0)
    >>> print(source1.csize, source2.csize)
    >>> combined = transform.ConcatenateSources(source1, source2, columns=['Position', 'Velocity'])
    >>> print(combined.csize)
    """
    from nbodykit.base.catalog import CatalogSource

    columns = kwargs.get('columns', None)
    if isinstance(columns, string_types):
        columns = [columns]

    # concatenate all columns, if none provided
    if columns is None or columns == []:
        columns = sources[0].columns

    # check comms
    if not all(src.comm == sources[0].comm for src in sources):
        raise ValueError("cannot concatenate sources: comm mismatch")

    # check all columns are there
    for source in sources:
        if not all(col in source for col in columns):
            raise ValueError(("cannot concatenate sources: columns are missing "
                              "from some sources"))
    # the total size
    size = numpy.sum([src.size for src in sources], dtype='intp')

    data = {}
    for col in columns:
        data[col] = da.concatenate([src[col] for src in sources], axis=0)

    toret = CatalogSource._from_columns(size, sources[0].comm, **data)
    toret.attrs.update(sources[0].attrs)
    return toret

def ConstantArray(value, size, chunks=100000):
    """
    Return a dask array of the specified ``size`` holding a single value.

    This uses numpy's "stride tricks" to avoid replicating
    the data in memory for each element of the array.

    Parameters
    ----------
    value : float
        the scalar value to fill the array with
    size : int
        the length of the returned dask array
    chunks : int, optional
        the size of the dask array chunks
    """
    ele = numpy.array(value)
    toret = numpy.lib.stride_tricks.as_strided(ele, [size] + list(ele.shape), [0] + list(ele.strides))
    return da.from_array(toret, chunks=chunks, name=False)


def CartesianToEquatorial(pos, observer=[0,0,0], frame='icrs'):
    """
    Convert Cartesian position coordinates to equatorial right ascension
    and declination, using the specified observer location.

    .. note::
        RA and DEC will be returned in degrees, with RA in the range [0,360]
        and DEC in the range [-90, 90].

    Parameters
    ----------
    pos : array_like
        a N x 3 array holding the Cartesian position coordinates
    observer : array_like
        a length 3 array holding the observer location
    frame : string
        A string, 'icrs' or 'galactic'. The frame of the input position.
        Use 'icrs' if the cartesian position is already in Equatorial.

    Returns
    -------
    ra, dec : array_like
        the right ascension and declination coordinates, in degrees. RA
        will be in the range [0,360] and DEC in the range [-90, 90]
    """

    # split x, y, z to signify that we do not need to have pos
    # as a full chunk in the last dimension.
    # this is useful when we use apply_gufunc.

    x, y, z = [pos[..., i] - observer[i] for i in range(3)]

    if frame == 'icrs':
        # FIXME: Convert these to a gufunc that uses astropy?
        # might be a step backward.

        # from equatorial to equatorial
        s = da.hypot(x, y)
        lon = da.arctan2(y, x)
        lat = da.arctan2(z, s)

        # convert to degrees
        lon = da.rad2deg(lon)
        lat = da.rad2deg(lat)
        # wrap lon to [0,360]
        lon = da.mod(lon-360., 360.)
        ra, dec = lon, lat
    else:
        from astropy.coordinates import SkyCoord

        def cart_to_eq(x, y, z):
            try:
                sc = SkyCoord(x, y, z, representation_type='cartesian', frame=frame)
                scg = sc.transform_to(frame='icrs')
                scg.representation_type = 'unitspherical'
            except:
                sc = SkyCoord(x, y, z, representation='cartesian', frame=frame)
                scg = sc.transform_to(frame='icrs')
                scg.representation = 'unitspherical'

            ra, dec = scg.ra.value, scg.dec.value

            return ra, dec

        dtype = pos.dtype
        ra, dec = da.apply_gufunc(cart_to_eq, '(),(),()->(),()', x, y, z, output_dtypes=[dtype, dtype])

    return da.stack((ra, dec), axis=0)

def CartesianToSky(pos, cosmo, velocity=None, observer=[0,0,0], zmax=100., frame='icrs'):
    r"""
    Convert Cartesian position coordinates to RA/Dec and redshift,
    using the specified cosmology to convert radial distances from
    the origin into redshift.

    If velocity is supplied, the returned redshift accounts for the
    additional peculiar velocity shift.

    Users should ensure that ``zmax`` is larger than the largest possible
    redshift being considered to avoid an interpolation exception.

    .. note::
        Cartesian coordinates should be in units of Mpc/h and velocity
        should be in units of km/s.

    Parameters
    ----------
    pos : dask array
        a N x 3 array holding the Cartesian position coordinates in Mpc/h
    cosmo : :class:`~nbodykit.cosmology.cosmology.Cosmology`
        the cosmology used to meausre the comoving distance from ``redshift``
    velocity : array_like
        a N x 3 array holding velocity in km/s
    observer : array_like, optional
        a length 3 array holding the observer location
    zmax : float, optional
        the maximum possible redshift, should be set to a reasonably large
        value to avoid interpolation failure going from comoving distance
        to redshift
    frame : string ('icrs' or 'galactic')
        speciefies which frame the Cartesian coordinates is. Useful if you know
        the simulation (usually cartesian) is in galactic units but you want
        to convert to the icrs (ra, dec) usually used in surveys.

    Returns
    -------
    ra, dec, z : dask array
        the right ascension (in degrees), declination (in degrees), and
        redshift coordinates. RA will be in the range [0,360] and DEC in the
        range [-90, 90]

    Notes
    -----
    If velocity is provided, redshift-space distortions are added to the
    real-space redshift :math:`z_\mathrm{real}`, via:

    .. math::

            z_\mathrm{redshift} = ( v_\mathrm{pec} / c ) (1 + z_\mathrm{reals})

    Raises
    ------
    TypeError
        If the input columns are not dask arrays
    """
    from astropy.constants import c
    from scipy.interpolate import interp1d

    if not isinstance(pos, da.Array):
        pos = da.from_array(pos, chunks=100000)

    pos = pos - observer
    # RA,dec coordinates (in degrees)
    ra, dec = CartesianToEquatorial(pos, frame=frame)

    # the distance from the origin
    r = da.linalg.norm(pos, axis=-1)

    def z_from_comoving_distance(x):
        zgrid = numpy.logspace(-8, numpy.log10(zmax), 1024)
        zgrid = numpy.concatenate([[0.], zgrid])
        rgrid = cosmo.comoving_distance(zgrid)
        return interp1d(rgrid, zgrid)(x)

    # invert distance - redshift relation
    z = r.map_blocks(z_from_comoving_distance)

    # add in velocity offsets?
    if velocity is not None:

        vpec = (pos * velocity).sum(axis=-1) / r

        z += vpec / c.to('km/s').value * (1 + z)

    return da.stack((ra, dec, z), axis=0)

def SkyToUnitSphere(ra, dec, degrees=True, frame='icrs'):
    """
    Convert sky coordinates (``ra``, ``dec``) to Cartesian coordinates on
    the unit sphere.

    Parameters
    ----------
    ra : :class:`dask.array.Array`; shape: (N,)
        the right ascension angular coordinate
    dec : :class:`dask.array.Array`; ; shape: (N,)
        the declination angular coordinate
    degrees : bool, optional
        specifies whether ``ra`` and ``dec`` are in degrees or radians
    frame : string ('icrs' or 'galactic')
        speciefies which frame the Cartesian coordinates is. Useful if you know
        the simulation (usually cartesian) is in galactic units but you want
        to convert to the icrs (ra, dec) usually used in surveys.

    Returns
    -------
    pos : :class:`dask.array.Array`; shape: (N,3)
        the cartesian position coordinates, where columns represent
        ``x``, ``y``, and ``z``

    Raises
    ------
    TypeError
        If the input columns are not dask arrays
    """
    ra, dec = da.broadcast_arrays(ra, dec)

    if frame == 'icrs':
        # no frame transformation
        # put into radians from degrees
        if degrees:
            ra  = da.deg2rad(ra)
            dec = da.deg2rad(dec)

        # cartesian coordinates
        x = da.cos( dec ) * da.cos( ra )
        y = da.cos( dec ) * da.sin( ra )
        z = da.sin( dec )
        return da.vstack([x,y,z]).T
    else:
        from astropy.coordinates import SkyCoord

        if degrees:
            ra  = da.deg2rad(ra)
            dec = da.deg2rad(dec)

        def eq_to_cart(ra, dec):
            try:
                sc = SkyCoord(ra, dec, unit='rad', representation_type='unitspherical', frame='icrs')
            except:
                sc = SkyCoord(ra, dec, unit='rad', representation='unitspherical', frame='icrs')

            scg = sc.transform_to(frame=frame)
            scg = scg.cartesian

            x, y, z = scg.x.value, scg.y.value, scg.z.value
            return numpy.stack([x, y, z], axis=1)

        arr = da.apply_gufunc(eq_to_cart, '(),()->(p)', ra, dec, output_dtypes=[ra.dtype], output_sizes={'p': 3})
        return arr

def SkyToCartesian(ra, dec, redshift, cosmo, observer=[0, 0, 0], degrees=True, frame='icrs'):
    """
    Convert sky coordinates (``ra``, ``dec``, ``redshift``) to a
    Cartesian ``Position`` column.

    .. warning::

        The returned Cartesian position is in units of Mpc/h.

    Parameters
    -----------
    ra : :class:`dask.array.Array`; shape: (N,)
        the right ascension angular coordinate
    dec : :class:`dask.array.Array`; shape: (N,)
        the declination angular coordinate
    redshift : :class:`dask.array.Array`; shape: (N,)
        the redshift coordinate
    cosmo : :class:`~nbodykit.cosmology.cosmology.Cosmology`
        the cosmology used to meausre the comoving distance from ``redshift``
    degrees : bool, optional
        specifies whether ``ra`` and ``dec`` are in degrees
    frame : string ('icrs' or 'galactic')
        speciefies which frame the Cartesian coordinates is. 

    Returns
    -------
    pos : :class:`dask.array.Array`; shape: (N,3)
        the cartesian position coordinates, where columns represent
        ``x``, ``y``, and ``z`` in units of Mpc/h

    Raises
    ------
    TypeError
        If the input columns are not dask arrays
    """
    ra, dec, redshift = da.broadcast_arrays(ra, dec, redshift)

    # pos on the unit sphere
    pos = SkyToUnitSphere(ra, dec, degrees=degrees, frame=frame)

    # multiply by the comoving distance in Mpc/h
    r = redshift.map_blocks(lambda z: cosmo.comoving_distance(z), dtype=redshift.dtype)

    return r[:,None] * pos + observer

def HaloConcentration(mass, cosmo, redshift, mdef='vir'):
    """
    Return halo concentration from halo mass, based on the analytic fitting
    formulas presented in
    `Dutton and Maccio 2014 <https://arxiv.org/abs/1402.7073>`_.

    .. note::
        The units of the input mass are assumed to be :math:`M_{\odot}/h`

    Parameters
    ----------
    mass : array_like
        either a numpy or dask array specifying the halo mass; units
        assumed to be :math:`M_{\odot}/h`
    cosmo : :class:`~nbodykit.cosmology.cosmology.Cosmology`
        the cosmology instance used in the analytic formula
    redshift : float
        compute the c(M) relation at this redshift
    mdef : str, optional
        string specifying the halo mass definition to use; should be
        'vir' or 'XXXc' or 'XXXm' where 'XXX' is an int specifying the
        overdensity

    Returns
    -------
    concen : :class:`dask.array.Array`
        a dask array holding the analytic concentration values

    References
    ----------
    Dutton and Maccio, "Cold dark matter haloes in the Planck era: evolution
    of structural parameters for Einasto and NFW profiles", 2014, arxiv:1402.7073

    """
    from halotools.empirical_models import NFWProfile

    mass, redshift = da.broadcast_arrays(mass, redshift)

    kws = {'cosmology':cosmo.to_astropy(), 'conc_mass_model':'dutton_maccio14', 'mdef':mdef}

    def get_nfw_conc(mass, redshift):
        kw1 = {}
        kw1.update(kws)
        kw1['redshift'] = redshift
        model = NFWProfile(**kw1)
        return model.conc_NFWmodel(prim_haloprop=mass)

    return da.map_blocks(get_nfw_conc, mass, redshift, dtype=mass.dtype)

def HaloVelocityDispersion(mass, cosmo, redshift, mdef='vir'):
    """ Compute the velocity dispersion of halo from Mass.

        This is a simple model suggested by Martin White.

        See http://adsabs.harvard.edu/abs/2008ApJ...672..122E
    """

    mass, redshift = da.broadcast_arrays(mass, redshift)
    def compute_vdisp(mass, redshift):
        h = cosmo.efunc(redshift)
        return 1100. * (h * mass / 1e15) ** 0.33333

    return da.map_blocks(compute_vdisp, mass, redshift, dtype=mass.dtype)

def HaloRadius(mass, cosmo, redshift, mdef='vir'):
    r"""
    Return proper halo radius from halo mass, based on the specified mass definition.
    This is independent of halo profile, and simply returns

    .. math::

        R = \left [ 3 M /(4\pi\Delta) \right]^{1/3}

    where :math:`\Delta` is the density threshold, which depends on cosmology,
    redshift, and mass definition

    .. note::
        The units of the input mass are assumed to be :math:`M_{\odot}/h`

    Parameters
    ----------
    mass : array_like
        either a numpy or dask array specifying the halo mass; units
        assumed to be :math:`M_{\odot}/h`
    cosmo : :class:`~nbodykit.cosmology.cosmology.Cosmology`
        the cosmology instance
    redshift : float
        compute the density threshold which determines the R(M) relation
        at this redshift
    mdef : str, optional
        string specifying the halo mass definition to use; should be
        'vir' or 'XXXc' or 'XXXm' where 'XXX' is an int specifying the
        overdensity

    Returns
    -------
    radius : :class:`dask.array.Array`
        a dask array holding the halo radius in 'physical Mpc/h [sic]'.
        This is proper Mpc/h, to convert to comoving, divide this by scaling factor.

    """
    from halotools.empirical_models import halo_mass_to_halo_radius

    mass, redshift = da.broadcast_arrays(mass, redshift)

    kws = {'cosmology':cosmo.to_astropy(), 'mdef':mdef}

    def mass_to_radius(mass, redshift):
        return halo_mass_to_halo_radius(mass=mass, redshift=redshift, **kws)

    return da.map_blocks(mass_to_radius, mass, redshift, dtype=mass.dtype)

# deprecated functions
vstack = deprecate("nbodykit.transform.vstack", StackColumns, "nbodykit.transform.StackColumns")
concatenate = deprecate("nbodykit.transform.concatenate", ConcatenateSources, "nbodykit.transform.ConcatenateSources")
SkyToCartesion = deprecate("nbodykit.transform.SkyToCartesion", SkyToCartesian, "nbodykit.transform.SkyToCartesian")
