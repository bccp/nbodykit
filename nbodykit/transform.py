import numpy
import dask.array as da
import dask

def ConstantArray(value, size, chunks=100000):
    """
    Return a dask array of the specified ``size`` holding a 
    single value
    
    This uses numpy's "stride tricks" to avoid replicating 
    the data in memory for each element of the array.
    
    Parameters
    ----------
    value : float
        the scalar value to fill the array with
    size : int
        the length of the returned dask array
    chunks : int; optional
        the size of the dask array chunks
    """
    toret = numpy.array(value) 
    toret = numpy.lib.stride_tricks.as_strided(toret, (size, toret.size), (0, toret.itemsize))
    return da.from_array(toret.squeeze(), chunks=chunks)
    

def SkyToUnitSphere(ra, dec, degrees=True):
    """
    Convert sky coordinates (ra, dec) coordinates to 
    cartesian coordinates on the unit sphere
    
    Parameters
    -----------
    ra, dec : dask.array, (N,)
        the input sky coordinates giving (ra, dec)
    degrees : bool, optional
        specifies whether ``ra`` and ``dec`` are in degrees
        
    Returns
    -------
    pos : dask.array, (N,3)
        the cartesian position coordinates, where columns represent 
        ``x``, ``y``, and ``z``
    """        
    # put into radians from degrees
    if degrees:
        ra  = da.deg2rad(ra)
        dec = da.deg2rad(dec)
    
    # cartesian coordinates
    x = da.cos( dec ) * da.cos( ra )
    y = da.cos( dec ) * da.sin( ra )
    z = da.sin( dec )        
    return da.vstack([x,y,z]).T
        
def SkyToCartesion(ra, dec, redshift, cosmo, degrees=True, interpolate_cdist=True):
    """
    Convert sky coordinates (ra, dec, redshift) coordinates to 
    cartesian coordinates, scaled to the comoving distance if `unit_sphere = False`, 
    else on the unit sphere
    
    
    Parameters
    -----------
    ra, dec, redshift : dask.array, (N,)
        the input sky coordinates giving (ra, dec, redshift)
    cosmo : astropy.cosmology.FLRW
        the cosmology used to meausre the comoving distance from ``redshift``
    degrees : bool, optional
        specifies whether ``ra`` and ``dec`` are in degrees
    interpolate_cdist : bool, optional
        if ``True``, interpolate the comoving distance as a function of redshift
        before evaluating the full results; can lead to significant speed improvements
        
    Returns
    -------
    pos : dask.array, (N,3)
        the cartesian position coordinates, where columns represent 
        ``x``, ``y``, and ``z``
    """
    # pos on the unit sphere
    pos = SkyToUnitSphere(ra, dec, degrees=degrees)
    
    # multiply by the comoving distance in Mpc/h
    if interpolate_cdist:
        comoving_distance = cosmo.comoving_distance.fit('z', bins=numpy.logspace(-5, 1, 1024))
    else:
        comoving_distance = cosmo.comoving_distance
    r = redshift.map_blocks(lambda z: comoving_distance(z).value * cosmo.h, dtype=redshift.dtype)
    
    return r[:,None] * pos
    
def HaloConcentration(mass, cosmo, redshift, mdef='vir'):
    """
    Return halo concentration from halo mass, based on the analytic fitting
    formulas presented in Dutton and Maccio 2014 (arxiv:1402.7073)
    
    .. note:: 
        The units of the input mass are assumed to be :math:`M_{\odot}/h`
    
    Parameters
    ----------
    mass : array_like
        either a numpy or dask array specifying the halo mass; units
        assumed to be :math:`M_{\odot}/h`
    cosmo : nbodykit.cosmology.Cosmology
        the cosmology instance used in the analytic formula
    redshift : float
        compute the c(M) relation at this redshift
    mdef : str; optional
        string specifying the halo mass definition to use; should be 
        'vir' or 'XXXc' or 'XXXm' where 'XXX' is an int specifying the 
        overdensity
    
    Returns
    -------
    concen : dask.array.Array
        a dask array holding the analytic concentration
    
    References
    ----------
    Dutton and Maccio, "Cold dark matter haloes in the Planck era: evolution 
    of structural parameters for Einasto and NFW profiles", 2014, arxiv:1402.7073
    """
    from halotools.empirical_models import ConcMass
    
    if not isinstance(mass, da.Array):
        mass = da.from_array(mass, chunks=100000)
    
    # initialize the model
    kws = {'cosmology':cosmo.engine, 'conc_mass_model':'dutton_maccio14', 'mdef':mdef, 'redshift':redshift}
    model = ConcMass(**kws)
            
    return mass.map_blocks(lambda mass: model.compute_concentration(prim_haloprop=mass), dtype=mass.dtype)
    
def HaloRadius(mass, cosmo, redshift, mdef='vir'):
    """
    Return halo radius from halo mass, based on the specified mass definition.
    This is independent of halo profile, and simply returns:
    
        ``radius = (mass * 3.0 / 4.0 / pi / rho)**(1.0 / 3.0)``
    
    where ``rho`` is the density threshold, which depends on cosmology, 
    redshift, and mass definition
    
    .. note:: 
        The units of the input mass are assumed to be :math:`M_{\odot}/h`
    
    Parameters
    ----------
    mass : array_like
        either a numpy or dask array specifying the halo mass; units
        assumed to be :math:`M_{\odot}/h`
    cosmo : nbodykit.cosmology.Cosmology
        the cosmology instance
    redshift : float
        compute the density threshold which determines the R(M) relation 
        at this redshift
    mdef : str; optional
        string specifying the halo mass definition to use; should be 
        'vir' or 'XXXc' or 'XXXm' where 'XXX' is an int specifying the 
        overdensity
    
    Returns
    -------
    radius : dask.array.Array
        a dask array holding the halo radius
    """
    from halotools.empirical_models import profile_helpers
    
    if not isinstance(mass, da.Array):
        mass = da.from_array(mass, chunks=100000)

    kws = {'cosmology':cosmo.engine, 'mdef':mdef, 'redshift':redshift}
    return mass.map_blocks(lambda mass: profile_helpers.halo_mass_to_halo_radius(mass=mass, **kws), dtype=mass.dtype)