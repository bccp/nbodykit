from pmesh.domain import GridND
from nbodykit.utils import split_size_3d
import numpy

def log_decomposition(comm, logger, N1, N2, pos1, pos2):
    """
    Log the distribution of particles to correlate across ranks that
    resulted after doing the domain decomposition.

    Parameters
    ----------
    comm :
        the MPI communicator
    logger :
        the current logger being used
    N1 : int
        the total number of objects in the first source being correlated
    N2 : int
        the total number of objects in the second source being correlated
    pos1 : array_like
        the domain-decomposed data of the first source
    pos2 : array_like
        the domain-decomposed data of the second source
    """
    # global counts
    if comm.rank == 0:
        logger.info('correlating %d x %d objects in total' %(N1, N2))

    # sizes for all ranks
    sizes1 = comm.gather(len(pos1), root=0)
    sizes2 = comm.gather(len(pos2), root=0)

    # rank 0 logs
    if comm.rank == 0:
        args = (numpy.median(sizes1), numpy.median(sizes2))
        logger.info("correlating A x B = %d x %d objects (median) per rank" % args)

        global_min = numpy.min(sizes1)
        logger.info("min A load per rank = %d" % global_min)

        global_max = numpy.max(sizes1)
        logger.info("max A load per rank = %d" % global_max)

        args = (N1//comm.size, N2)
        logger.info("(even distribution would result in %d x %d)" % args)

def decompose_box_data(first, second, attrs, logger, smoothing):
    """
    Perform a domain decomposition on simulation box data, returning the
    domain-demposed position and weight arrays for each object in the
    correlating pair.

    No load balancing is required since the particles in are assumed to
    be in a box.

    The implementation follows:

    1. Decompose the first source such that the objects are spatially
       tight on a given rank.
    2. Decompose the second source, ensuring a given rank holds all
       particles within the desired maximum separation.

    Parameters
    ----------
    first : CatalogSource
        the first source we are correlating
    second : CatalogSource
        the second source we are correlating
    attrs : dict
        dict of parameters from the pair counting algorithm
    logger :
        the current active logger
    smoothing :
        the maximum Cartesian separation implied by the user's binning

    Returns
    -------
    (pos1, w1), (pos2, w2) : array_like
        the (decomposed) set of positions and weights to correlate
    """
    comm = first.comm

    # determine processor division for domain decomposition
    np = split_size_3d(comm.size)
    if comm.rank == 0:
        logger.info("using cpu grid decomposition: %s" %str(np))

    # get the (periodic-enforced) position for first
    pos1 = first['Position']
    if attrs['periodic']:
        pos1 %= attrs['BoxSize']
    pos1, w1 = first.compute(pos1, first[attrs['weight']])
    N1 = comm.allreduce(len(pos1))

    # get the (periodic-enforced) position for second
    if second is not None:
        pos2 = second['Position']
        if attrs['periodic']:
            pos2 %= attrs['BoxSize']
        pos2, w2 = second.compute(pos2, second[attrs['weight']])
        N2 = comm.allreduce(len(pos2))
    else:
        pos2 = pos1
        w2 = w1
        N2 = N1

    # domain decomposition
    grid = [
        numpy.linspace(0, attrs['BoxSize'][0], np[0] + 1, endpoint=True),
        numpy.linspace(0, attrs['BoxSize'][1], np[1] + 1, endpoint=True),
        numpy.linspace(0, attrs['BoxSize'][2], np[2] + 1, endpoint=True),
    ]
    domain = GridND(grid, comm=comm)

    # exchange first particles
    layout = domain.decompose(pos1, smoothing=0)
    pos1 = layout.exchange(pos1)
    w1 = layout.exchange(w1)

    # exchange second particles
    if smoothing > attrs['BoxSize'].max() * 0.25:
        pos2 = numpy.concatenate(comm.allgather(pos2), axis=0)
        w2   = numpy.concatenate(comm.allgather(w2), axis=0)
    else:
        layout  = domain.decompose(pos2, smoothing=smoothing)
        pos2 = layout.exchange(pos2)
        w2   = layout.exchange(w2)

    # log the decomposition breakdown
    log_decomposition(comm, logger, N1, N2, pos1, pos2)

    return (pos1, w1), (pos2, w2)


def decompose_survey_data(first, second, attrs, logger, smoothing, domain_factor=2,
                            angular=False, return_cartesian=False):
    """
    Perform a domain decomposition on survey data, returning the
    domain-demposed position and weight arrays for each object in the
    correlating pair.

    The domain decomposition is based on the Cartesian coordinates of
    the input data (assumed to be in sky coordinates).

    Load balancing is required since the distribution in Cartesian space
    will likely not be uniform.

    The implementation follows:

    1. Decompose the first source and balance the particle load, such that
       the first source is evenly distributed across all ranks and the
       objects are spatially tight on a given rank.
    2. Decompose the second source, ensuring a given rank holds all
       particles within the desired maximum separation.

    Parameters
    ----------
    first : CatalogSource
        the first source we are correlating
    second : CatalogSource
        the second source we are correlating
    attrs : dict
        dict of parameters from the pair counting algorithm
    logger :
        the current active logger
    smoothing :
        the maximum Cartesian separation implied by the user's binning
    domain_factor : int, optional
        the factor by which we over-sample the mesh with cells in a given
        direction; higher values can lead to better performance
    angular : bool, optional
        if ``True``, the Cartesian positions used in the domain
        decomposition are on the unit sphere
    return_cartesian : bool, optional
        whether to return the pos as (ra, dec, z), or the Cartesian (x, y, z)

    Returns
    -------
    (pos1, w1), (pos2, w2) : array_like
        the (decomposed) set of positions and weights to correlate
    """
    from nbodykit.transform import StackColumns
    comm = first.comm

    # either (ra,dec) or (ra,dec,redshift)
    poscols = [attrs['ra'], attrs['dec']]
    if not angular: poscols += [attrs['redshift']]

    # determine processor division for domain decomposition
    np = split_size_3d(comm.size)
    if comm.rank == 0:
        logger.info("using cpu grid decomposition: %s" %str(np))

    # stack position and compute
    pos1 = StackColumns(*[first[col] for col in poscols])
    pos1, w1 = first.compute(pos1, first[attrs['weight']])
    N1 = comm.allreduce(len(pos1))

    # only need cosmo if not angular
    cosmo = attrs.get('cosmo', None) if not angular else None
    if not angular and cosmo is None:
        raise ValueError("need a cosmology to decompose non-angular survey data")
    cpos1, cpos1_min, cpos1_max, rdist1 = get_cartesian(comm, pos1, cosmo=cosmo)

    # pass in comoving dist to Corrfunc instead of redshift
    if not angular:
        pos1 = pos1.copy() # we need to overwrite it; dask doesn't always return a copy after 0.18.1
        pos1[:,2] = rdist1

    # set up position for second too
    if second is not None:

        # stack position and compute for "second"
        pos2 = StackColumns(*[second[col] for col in poscols])
        pos2, w2 = second.compute(pos2, second[attrs['weight']])
        N2 = comm.allreduce(len(pos2))

        # get comoving dist and boxsize
        cpos2, cpos2_min, cpos2_max, rdist2 = get_cartesian(comm, pos2, cosmo=cosmo)

        # pass in comoving distance instead of redshift
        if not angular:
            pos2 = pos2.copy() # we need to overwrite it; dask doesn't always return a copy after 0.18.1
            pos2[:,2] = rdist2
    else:
        pos2 = pos1
        w2 = w1
        N2 = N1
        cpos2_min = cpos1_min
        cpos2_max = cpos1_max
        cpos2 = cpos1

    # determine global boxsize
    if second is None:
        cpos_min = cpos1_min
        cpos_max = cpos1_max
    else:
        cpos_min = numpy.min(numpy.vstack([cpos1_min, cpos2_min]), axis=0)
        cpos_max = numpy.max(numpy.vstack([cpos1_max, cpos2_max]), axis=0)

    boxsize = cpos_max - cpos_min

    if comm.rank == 0:
        logger.info("position variable range on rank 0 (max, min) = %s, %s" % (cpos_max, cpos_min))

    # initialize the domain
    # NOTE: over-decompose by factor of 2 to trigger load balancing
    grid = [
        numpy.linspace(cpos_min[0], cpos_max[0], domain_factor*np[0] + 1, endpoint=True),
        numpy.linspace(cpos_min[1], cpos_max[1], domain_factor*np[1] + 1, endpoint=True),
        numpy.linspace(cpos_min[2], cpos_max[2], domain_factor*np[2] + 1, endpoint=True),
    ]
    domain = GridND(grid, comm=comm, periodic=False)

    # balance the load
    domain.loadbalance(domain.load(cpos1))

    if comm.rank == 0:
        logger.info("Load balance done")

    # if we want to return cartesian, redefine pos
    if return_cartesian:
        pos1 = cpos1
        pos2 = cpos2

    # decompose based on cartesian positions
    layout = domain.decompose(cpos1, smoothing=0)
    pos1   = layout.exchange(pos1)
    w1     = layout.exchange(w1)

    # get the position/weight of the secondaries
    if smoothing > boxsize.max() * 0.25:
        pos2 = numpy.concatenate(comm.allgather(pos2), axis=0)
        w2   = numpy.concatenate(comm.allgather(w2), axis=0)
    else:
        layout  = domain.decompose(cpos2, smoothing=smoothing)
        pos2 = layout.exchange(pos2)
        w2   = layout.exchange(w2)

    # log the decomposition breakdown
    log_decomposition(comm, logger, N1, N2, pos1, pos2)

    return (pos1, w1), (pos2, w2)

def get_cartesian(comm, pos, cosmo=None):
    """
    Utility function to convert sky coordinates to Cartesian coordinates and
    return the implied box size from the position bounds.

    If ``cosmo`` is not provided, return coordinates on the unit sphere.
    """
    from nbodykit.utils import get_data_bounds

    # get RA,DEC in degrees
    ra, dec = numpy.deg2rad(pos[:,0]), numpy.deg2rad(pos[:,1])

    # cartesian position
    x = numpy.cos( dec ) * numpy.cos( ra )
    y = numpy.cos( dec ) * numpy.sin( ra )
    z = numpy.sin( dec )
    cpos = numpy.vstack([x,y,z]).T

    # multiply by comoving distance?
    if cosmo is not None:
        assert pos.shape[-1] == 3
        rdist = cosmo.comoving_distance(pos[:,2]) # in Mpc/h
        cpos = rdist[:,None] * cpos
    else:
        rdist = None

    # min/max of position
    cpos_min, cpos_max = get_data_bounds(cpos, comm)
    boxsize = abs(cpos_max - cpos_min)

    # some padding to avoid weird effects with domain decomposition
    # like sitting on an edge and goes out of bound due to round off errors.

    return cpos, cpos_min - 1e-3 * boxsize, cpos_max + 1e-3 * boxsize, rdist
