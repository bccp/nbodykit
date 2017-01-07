import numpy
import dask.array as da
import dask

def DefaultSelection(source):
    return da.ones(len(source), dtype='?', chunks=100000)

def DefaultWeight(source):
    return da.ones(len(source), dtype='f4', chunks=100000)

def SkyToCartesion(ra, dec, redshift, degrees=True, cosmo=None, unit_sphere=False):
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
    
    """
    Convert sky coordinates (ra, dec, redshift) coordinates to 
    cartesian coordinates, scaled to the comoving distance if `unit_sphere = False`, 
    else on the unit sphere
    
    
    Parameters
    -----------
    ra, dec, redshift : dask.array, (N,)
        the input sky coordinates giving (ra, dec, redshift)
    degrees : bool, optional
        specifies whether ``ra`` and ``dec`` are in degrees
    cosmo : astropy.cosmology.FLRW, optional
        the cosmology used to meausre the comoving distance from ``redshift``;
        required if ``unit_sphere=False``
    unit_sphere : bool, optional
        if True, use a comoving distance of unity for all objects
        
    Returns
    -------
    pos : dask.array, (N,3)
        the cartesian position coordinates, where columns represent 
        ``x``, ``y``, and ``z``
    """
    # put into radians from degrees
    if degrees:
        ra = da.deg2rad(ra)
        dec = da.deg2rad(dec)
    
    # cartesian coordinates
    x = da.cos( dec ) * da.cos( ra )
    y = da.cos( dec ) * da.sin( ra )
    z = da.sin( dec )        
    pos = da.vstack([x,y,z])
    
    # multiply by the comoving distance in Mpc/h
    if not unit_sphere:
        r = redshift.map_blocks(lambda z: cosmo.comoving_distance(z).value * cosmo.h, dtype=redshift.dtype)
        pos *= r
    
    return pos.T

def PackXYZ(source):
    return da.vstack([source['x'], source['y'], source['z']]).T

