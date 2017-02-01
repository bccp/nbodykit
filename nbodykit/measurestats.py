from mpi4py import MPI
import numpy
import logging
import time

from nbodykit.core import Painter, Transfer
from nbodykit.utils.meshtools import SlabIterator
from nbodykit.utils import timer

logger = logging.getLogger('measurestats')

#------------------------------------------------------------------------------
# Fourier space statistics
#------------------------------------------------------------------------------
def compute_3d_power(fields, pm, comm=None):
    """
    Compute and return the 3D power from two input fields

    Parameters
    ----------
    fields : list of tuples of (DataSource, Painter, Transfer)
        the list of fields which the 3D power will be computed
    pm : ParticleMesh
        the particle mesh object that handles the painting and FFTs
    comm : MPI.Communicator, optional
        the communicator to pass to the ParticleMesh object. If not
        provided, ``MPI.COMM_WORLD`` is used

    Returns
    -------
    p3d : array_like (complex)
        the 3D complex array holding the power spectrum
    stats1 : dict
        statistics of the first field, as returned by the `Painter` 
    stats2 : dict
        statistics of the second field, as returned by the `Painter`
    """
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank

    # extract lists of datasources, painters, and transfers separately
    datasources = [d for d, p, t in fields]
    painters    = [p for d, p, t in fields]
    transfers   = [t for d, p, t in fields]

    # step 1: paint the density field to the mesh
    real, stats1 = painters[0].paint(pm, datasources[0])
    if rank == 0: logger.info('%s painting done' %painters[0].paintbrush)

    # step 2: Fourier transform density field using real to complex FFT
    complex = real.r2c()
    del real

    if rank == 0: logger.info('r2c done')

    # step 3: apply transfer function kernels to complex field
    for t in transfers[0]: t(pm, complex)

    # compute the auto power of single supplied field
    if len(fields) == 1:
        c1 = complex
        c2 = complex
        stats2 = stats1

    # compute the cross power of the two supplied fields
    else:
        c1 = complex

        # crash if box size isn't the same
        if not numpy.all(datasources[0].BoxSize == datasources[1].BoxSize):
            raise ValueError("mismatch in box sizes for cross power measurement")

        # apply painting, FFT, and transfer steps to second field
        real, stats2 = painters[1].paint(pm, datasources[1])
        if rank == 0: logger.info('%s painting 2 done' %painters[1].paintbrush)
        c2 = real.r2c()
        del real
        if rank == 0: logger.info('r2c 2 done')

        for t in transfers[1]: t(pm, c2)

    # calculate the 3d power spectrum, slab-by-slab to save memory
    p3d = c1
    for (s0, s1, s2) in zip(p3d.slabs, c1.slabs, c2.slabs):
        s0[...] = s1 * s2.conj()

    # the complex field is dimensionless; power is L^3
    # ref to http://icc.dur.ac.uk/~tt/Lectures/UA/L4/cosmology.pdf
    p3d[...] *= pm.BoxSize.prod() 

    return p3d, stats1, stats2

def apply_bianchi_kernel(data, x3d, i, j, k=None):
    """
    Apply coordinate kernels to ``data`` necessary to compute the 
    power spectrum multipoles via FFTs using the algorithm 
    detailed in Bianchi et al. 2015.
    
    This multiplies by one of two kernels:
    
        1. x_i * x_j / x**2 * data, if `k` is None
        2. x_i**2 * x_j * x_k / x**4 * data, if `k` is not None
    
    See equations 10 (for quadrupole) and 12 (for hexadecapole)
    of Bianchi et al 2015.
    
    Parameters
    ----------
    data : array_like
        the array to rescale -- either the configuration-space 
        `pm.real` or the Fourier-space `pm.complex`
    x : array_like
        the coordinate array -- either `pm.r` or `pm.k`
    i, j, k : int
        the integers specifying the coordinate axes; see the 
        above description 
    """        
    # loop over yz slabs of the mesh
    for slab in SlabIterator(x3d, axis=0):
    
        # normalization is norm squared of coordinate mesh
        norm = slab.norm2()
                    
        # get coordinate arrays for indices i, j         
        xi = slab.coords(i)
        if j == i: xj = xi
        else: xj = slab.coords(j)
            
        # handle third index j
        if k is not None:
            
            # get coordinate array for index k
            if k == i: xk = xi
            elif k == j: xk = xj
            else: xk = slab.coords(k)
            
            # weight data by xi**2 * xj * xj 
            data[slab.index] = data[slab.index] * xi**2 * xj * xk / norm**2
            data[slab.index][norm==0] = 0.
        else:
            # weight data by xi * xj
            data[slab.index] = data[slab.index] * xi * xj / norm
            data[slab.index][norm==0] = 0.
                
def compute_bianchi_poles(comm, max_ell, catalog, Nmesh, factor_hexadecapole=False, paintbrush='cic'):
    """
    Use the algorithm detailed in Bianchi et al. 2015 to compute and return the 3D 
    power spectrum multipoles (`ell = [0, 2, 4]`) from one input field, which contains 
    non-trivial survey geometry.
    
    The estimator uses the FFT algorithm outlined in Bianchi et al. 2015
    (http://adsabs.harvard.edu/abs/2015arXiv150505341B) to compute
    the monopole, quadrupole, and hexadecapole
    
    Parameters
    ----------
    comm : MPI.Communicator
        the communicator to pass to the ParticleMesh object
    max_ell : int, {0, 2, 4}
        the maximum multipole number to compute up to (inclusive)
    catalog : :class:`~nbodykit.fkp.FKPCatalog`
        the FKP catalog object that manipulates the data and randoms DataSource
        objects and paints the FKP density
    Nmesh : int
        the number of cells (per side) in the gridded mesh
    factor_hexadecapole : bool, optional
        if `True`, use the factored expression for the hexadecapole (ell=4) from
        eq. 27 of Scoccimarro 2015 (1506.02729); default is `False`
    paintbrush : str, {'cic', 'tsc'}
        the density assignment kernel to use when painting, either `cic` (2nd order) 
        or `tsc` (3rd order); default is `cic`
    
    Returns
    -------
    pm : ParticleMesh
        the mesh object used to do painting, FFTs, etc
    result : list of arrays
        list of 3D complex arrays holding power spectrum multipoles; respectively, 
        if `ell_max=0,2,4`, the list holds the monopole only, monopole and quadrupole, 
        or the monopole, quadrupole, and hexadecapole
    stats : dict
        dict holding the statistics of the input fields, as returned
        by the `FKPPainter` painter
    
    References
    ----------
    * Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
      MNRAS, 2015
    * Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
    """
    from pmesh.pm import ParticleMesh, RealField
    
    rank = comm.rank
    bianchi_transfers = []

    # the painting kernel transfer
    if paintbrush == 'cic':
        transfer = Transfer.create('AnisotropicCIC')
    elif paintbrush == 'tsc':
        transfer = Transfer.create('AnisotropicTSC')
    else:
        raise ValueError("valid `paintbrush` values are: ['cic', 'tsc']")

    # determine which multipole values we are computing
    if max_ell not in [0, 2, 4]:
        raise ValueError("valid values for the maximum multipole number are [0, 2, 4]")
    ells = numpy.arange(0, max_ell+1, 2)
    
    # determine kernels needed to compute quadrupole
    if max_ell > 0:
        
        # the (i,j) index values for each kernel
        k2 = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        
        # the amplitude of each kernel term
        a2 = [1.]*3 + [2.]*3
        bianchi_transfers.append((a2, k2))
    
    # determine kernels needed to compute hexadecapole
    if max_ell > 2 and not factor_hexadecapole:
        
        # the (i,j,k) index values for each kernel
        k4 = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2),
             (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 1), (0, 1, 1),
             (0, 2, 2), (1, 2, 2), (0, 1, 2), (1, 0, 2), (2, 0, 1)]
        
        # the amplitude of each kernel term
        a4 = [1.]*3 + [4.]*6 + [6.]*3 + [12.]*3
        bianchi_transfers.append((a4, k4))
    
    # load the data/randoms, setup boxsize, etc from the FKPCatalog
    with catalog:
        
        # the mean coordinate offset of the input data
        offset = catalog.mean_coordinate_offset
        
        # initialize the particle mesh
        pm = ParticleMesh(BoxSize=catalog.BoxSize, Nmesh=[Nmesh]*3, dtype='f4', comm=comm)
        
        # paint the FKP density field to the mesh (paints: data - randoms, essentially)
        real, stats = catalog.paint(pm, paintbrush=paintbrush)

    # save the painted density field for later
    density = real.copy()
    if rank == 0: logger.info('%s painting done' %paintbrush)
    
    # FFT density field and apply the paintbrush window transfer kernel
    complex = real.r2c()
    transfer(pm, complex)
    if rank == 0: logger.info('ell = 0 done; 1 r2c completed')
        
    # monopole A0 is just the FFT of the FKP density field
    volume = pm.BoxSize.prod()
    A0 = complex[:]*volume # normalize with a factor of volume
    
    # store the A0, A2, A4 arrays here
    result = []
    result.append(A0)
    
    # the real-space grid points
    # the grid is properly offset from [-L/2, L/2] to the original positions in space
    # this is the grid used when applying Bianchi kernels
    cell_size = pm.BoxSize / pm.Nmesh
    xgrid = [(ri+0.5)*cell_size[i] + offset[i] for i, ri in enumerate(pm.r)]
    
    # loop over the higher order multipoles (ell > 0)
    start = time.time()
    for iell, ell in enumerate(ells[1:]):
        
        # temporary array to hold sum of all of the terms in Fourier space
        Aell_sum = numpy.zeros_like(complex)
        
        # loop over each kernel term for this multipole
        for amp, integers in zip(*bianchi_transfers[iell]):
                        
            # reset the realspace mesh to the original FKP density
            real[:] = density[:]
        
            # apply the real-space Bianchi kernel
            if rank == 0: logger.debug("applying real-space Bianchi transfer for %s..." %str(integers))
            apply_bianchi_kernel(real, xgrid, *integers)
            if rank == 0: logger.debug('...done')
    
            # do the real-to-complex FFT
            if rank == 0: logger.debug("performing r2c...")
            real.r2c(out=complex)
            if rank == 0: logger.debug('...done')
            
            # apply the Fourier-space Bianchi kernel
            if rank == 0: logger.debug("applying Fourier-space Bianchi transfer for %s..." %str(integers))
            apply_bianchi_kernel(complex, pm.k, *integers)
            if rank == 0: logger.debug('...done')
            
            # and this contribution to the total sum
            Aell_sum[:] += amp*complex[:]*volume
            
        # apply the paintbrush window transfer function and save
        transfer(pm, Aell_sum)
        result.append(Aell_sum); del Aell_sum # delete temp array since appending to list makes copy
        
        # log the total number of FFTs computed for each ell
        if rank == 0: 
            args = (ell, len(bianchi_transfers[iell][0]))
            logger.info('ell = %d done; %s r2c completed' %args)
        
    # density array no longer needed
    del density
    
    # summarize how long it took
    stop = time.time()
    if rank == 0:
        logger.info("higher order multipoles computed in elapsed time %s" %timer(start, stop))
        if factor_hexadecapole:
            logger.info("using factorized hexadecapole estimator for ell=4")
    
    # proper normalization: same as equation 49 of Scoccimarro et al. 2015 
    norm = 1.0 / stats['A_ran']
    
    # reuse memory for output arrays
    P0 = result[0]
    if max_ell > 0: 
        P2 = result[1]
    if max_ell > 2:
        P4 = numpy.empty_like(P2) if factor_hexadecapole else result[2]
        
    # calculate the power spectrum multipoles, slab-by-slab to save memory
    for islab in range(len(P0)):

        # save arrays for reuse
        P0_star = (P0[islab]).conj()
        if max_ell > 0: P2_star = (P2[islab]).conj()

        # hexadecapole
        if max_ell > 2:
            
            # see equation 8 of Bianchi et al. 2015
            if not factor_hexadecapole:
                P4[islab, ...] = norm * 9./8. * P0[islab] * (35.*(P4[islab]).conj() - 30.*P2_star + 3.*P0_star)
            # see equation 48 of Scoccimarro et al; 2015
            else:
                P4[islab, ...] = norm * 9./8. * ( 35.*P2[islab]*P2_star + 3.*P0[islab]*P0_star - 5./3.*(11.*P0[islab]*P2_star + 7.*P2[islab]*P0_star) )
        
        # quadrupole: equation 7 of Bianchi et al. 2015
        if max_ell > 0:
            P2[islab, ...] = norm * 5./2. * P0[islab] * (3.*P2_star - P0_star)

        # monopole: equation 6 of Bianchi et al. 2015
        P0[islab, ...] = norm * P0[islab] * P0_star
        
    return pm, result, stats

#------------------------------------------------------------------------------
# configuration space statistics
#------------------------------------------------------------------------------
def compute_brutal_corr(datasources, redges, Nmu=0, comm=None, subsample=1, los='z', poles=[]):
    r"""
    Compute the correlation function by direct pair summation, either as a function
    of separation (`R`) or as a function of separation and line-of-sight angle (`R`, `mu`)
    
    The estimator used to compute the correlation function is:
    
    .. math:: 
        
        \xi(r, \mu) = DD(r, \mu) / RR(r, \mu) - 1.
    
    where `DD` is the number of data-data pairs, and `RR` is the number of random-random pairs,
    which is determined solely by the binning used, assuming a constant number density
    
    Parameters
    ----------
    datasources : list of DataSource objects
        the list of data instances from which the 3D correlation will be computed
    redges : array_like
        the bin edges for the `R` variable
    Nmu : int, optional
        the number of desired `mu` bins, where `mu` is the cosine 
        of the angle from the line-of-sight. Default is `0`, in 
        which case the correlation function is binned as a function of `R` only
    comm : MPI.Communicator, optional
        the communicator to pass to the ``ParticleMesh`` object. If not
        provided, ``MPI.COMM_WORLD`` is used
    subsample : int, optional
        downsample the input datasources by choosing 1 out of every `N` points. 
        Default is `1` (no subsampling).
    los : str, {'x', 'y', 'z'}, optional
        the dimension to treat as the line-of-sight; default is 'z'.
    poles : list of int, optional
        integers specifying the multipoles to compute from the 2D correlation function
        
    Returns
    -------
    pc : :class:`kdcount.correlate.paircount`
        the pair counting instance 
    xi : array_like
        the correlation function result; if `poles` supplied, the shape is 
        `(len(redges)-1, len(poles))`, otherwise, the shape is either `(len(redges)-1, )`
        or `(len(redges)-1, Nmu)`
    RR : array_like
        the number of random-random pairs (used as normalization of the data-data pairs)
    """
    from pmesh.domain import GridND
    from kdcount import correlate
    
    # some setup
    if los not in "xyz": raise ValueError("`los` must be `x`, `y`, or `z`")
    los   = "xyz".index(los)
    poles = numpy.array(poles)
    Rmax  = redges[-1]
    if comm is None: comm = MPI.COMM_WORLD
    
    # determine processor division for domain decomposition
    for Nx in range(int(comm.size**0.3333) + 1, 0, -1):
        if comm.size % Nx == 0: break
    else:
        Nx = 1
    for Ny in range(int(comm.size**0.5) + 1, 0, -1):
        if (comm.size // Nx) % Ny == 0: break
    else:
        Ny = 1
    Nz = comm.size // Nx // Ny
    Nproc = [Nx, Ny, Nz]
    
    # log some info
    if comm.rank == 0:
        logger.info('Nproc = %s' %str(Nproc))
        logger.info('Rmax = %g' %Rmax)
    
    # domain decomposition
    grid = [numpy.linspace(0, datasources[0].BoxSize[i], Nproc[i]+1, endpoint=True) for i in range(3)]
    domain = GridND(grid, comm=comm)

    # read position for field #1 
    with datasources[0].open() as stream:
        [[pos1]] = stream.read(['Position'], full=True)
    pos1 = pos1[comm.rank * subsample // comm.size ::subsample]
    N1 = comm.allreduce(len(pos1))
    
    # read position for field #2
    if len(datasources) > 1:
        with datasources[1].open() as stream:
            [[pos2]] = stream.read(['Position'], full=True)
        pos2 = pos2[comm.rank * subsample // comm.size ::subsample]
        N2 = comm.allreduce(len(pos2))
    else:
        pos2 = pos1
        N2 = N1
    
    # exchange field #1 positions    
    layout = domain.decompose(pos1, smoothing=0)
    pos1 = layout.exchange(pos1)
    if comm.rank == 0: logger.info('exchange pos1')
        
    # exchange field #2 positions
    if Rmax > datasources[0].BoxSize[0] * 0.25:
        pos2 = numpy.concatenate(comm.allgather(pos2), axis=0)
    else:
        layout = domain.decompose(pos2, smoothing=Rmax)
        pos2 = layout.exchange(pos2)
    if comm.rank == 0: logger.info('exchange pos2')

    # initialize the trees to hold the field points
    tree1 = correlate.points(pos1, boxsize=datasources[0].BoxSize)
    tree2 = correlate.points(pos2, boxsize=datasources[0].BoxSize)

    # log the sizes of the trees
    logger.info('rank %d correlating %d x %d' %(comm.rank, len(tree1), len(tree2)))
    if comm.rank == 0: logger.info('all correlating %d x %d' %(N1, N2))

    # use multipole binning
    if len(poles):
        bins = correlate.FlatSkyMultipoleBinning(redges, poles, los)
    # use (R, mu) binning
    elif Nmu > 0:
        bins = correlate.FlatSkyBinning(redges, Nmu, los)
    # use R binning
    else:
        bins = correlate.RBinning(redges)

    # do the pair counting
    # have to set usefast = False to get mean centers, or exception thrown
    pc = correlate.paircount(tree2, tree1, bins, np=0, usefast=False, compute_mean_coords=True)
    pc.sum1[:] = comm.allreduce(pc.sum1)
    
    # get the mean bin values, reducing from all ranks
    pc.pair_counts[:] = comm.allreduce(pc.pair_counts)
    with numpy.errstate(invalid='ignore'):
        if bins.Ndim > 1:
            for i in range(bins.Ndim):
                pc.mean_centers[i][:] = comm.allreduce(pc.mean_centers_sum[i]) / pc.pair_counts
        else:
            pc.mean_centers[:] = comm.allreduce(pc.mean_centers_sum[0]) / pc.pair_counts

    # compute the random pairs from the fractional volume
    RR = 1.*N1*N2 / datasources[0].BoxSize.prod()
    if Nmu > 0:
        dr3 = numpy.diff(pc.edges[0]**3)
        dmu = numpy.diff(pc.edges[1])
        RR *= 2. / 3. * numpy.pi * dr3[:,None] * dmu[None,:]
    else:
        RR *= 4. / 3. * numpy.pi * numpy.diff(pc.edges**3)
    
    # return the correlation and the pair count object
    xi = (1. * pc.sum1 / RR) - 1.0
    if len(poles):
        xi = xi.T # makes ell the second axis 
        xi[:,poles!=0] += 1.0 # only monopole gets the minus one

    return pc, xi, RR

def compute_3d_corr(fields, pm, comm=None):
    """
    Compute the 3D correlation function by Fourier transforming 
    the 3D power spectrum.

    See the documentation for :func:`compute_3d_power` for details
    of input parameters and return types.
    """
    # the 3D power spectrum
    p3d, stats1, stats2 = compute_3d_power(fields, pm, comm=comm)

    # directly transform dimensionless p3d
    # Note that L^3 cancels with dk^3.
    p3d[...] *= 1.0 / pm.BoxSize.prod()
    xi3d = p3d.c2r()

    return xi3d, stats1, stats2

#------------------------------------------------------------------------------
# general tools
#------------------------------------------------------------------------------
def project_to_basis(comm, x3d, y3d, edges, los='z', poles=[], hermitian_symmetric=False):
    """ 
    Project a 3D statistic on to the specified basis. The basis will be one 
    of:
    
        - 2D (`x`, `mu`) bins: `mu` is the cosine of the angle to the line-of-sight
        - 2D (`x`, `ell`) bins: `ell` is the multipole number, which specifies
          the Legendre polynomial when weighting different `mu` bins
    
    .. note::
    
        The 2D (`x`, `mu`) bins will be computed only if `poles` is specified.
        See return types for further details.
    
    Notes
    -----
    *   the `mu` range extends from 0.0 to 1.0
    *   the `mu` bins are half-inclusive half-exclusive, except the last bin
        is inclusive on both ends (to include `mu = 1.0`)
    
    Parameters
    ----------
    comm : MPI.Communicatior
        the MPI communicator
    x3d  : list of array_like
        list of coordinate values for each dimension of the `y3d` array; the 
        items must able to be broadcasted to the same shape as `y3d`
    y3d : array_like, real or complex
        the 3D array holding the statistic to be projected to the specified basis
    edges : list of arrays, (2,)
        list of arrays specifying the edges of the desired `x` bins and `mu` bins
    los : str, {'x','y','z'}, optional
        the line-of-sight direction to use, which `mu` is defined with
        respect to; default is `z`.
    poles : list of int, optional
        if provided, a list of integers specifying multipole numbers to
        project the 2d `(x, mu)` bins on to  
    hermitian_symmetric : bool, optional
        Whether the input array `y3d` is Hermitian-symmetric, i.e., the negative 
        frequency terms are just the complex conjugates of the corresponding 
        positive-frequency terms; if ``True``, the positive frequency terms
        will be explicitly double-counted to account for this symmetry
    
    Returns
    -------
    result : tuple
        the 2D binned results; a tuple of ``(xmean_2d, mumean_2d, y2d, N_2d)``, where:
        
        xmean_2d : array_like, (Nx, Nmu)
            the mean `x` value in each 2D bin
        mumean_2d : array_like, (Nx, Nmu)
            the mean `mu` value in each 2D bin
        y2d : array_like, (Nx, Nmu)
            the mean `y3d` value in each 2D bin
        N_2d : array_like, (Nx, Nmu)
            the number of values averaged in each 2D bin
    
    pole_result : tuple or `None`
        the multipole results; if `poles` supplied it is a tuple of ``(xmean_1d, poles, N_1d)``, 
        where:
    
        xmean_1d : array_like, (Nx,)
            the mean `x` value in each 1D multipole bin
        poles : array_like, (Nell, Nx)
            the mean multipoles value in each 1D bin
        N_1d : array_like, (Nx,)
            the number of values averaged in each 1D bin
    """
    from scipy.special import legendre

    # setup the bin edges and number of bins
    xedges, muedges = edges
    x2edges = xedges**2
    Nx = len(xedges) - 1 
    Nmu = len(muedges) - 1
    
    # always make sure first ell value is monopole, which
    # is just (x, mu) projection since legendre of ell=0 is 1
    do_poles = len(poles) > 0
    _poles = [0]+sorted(poles) if 0 not in poles else sorted(poles)
    legpoly = [legendre(l) for l in _poles]
    ell_idx = [_poles.index(l) for l in poles]
    Nell = len(_poles)
    
    # valid ell values
    if any(ell < 0 for ell in _poles):
        raise ValueError("in `project_to_basis`, multipole numbers must be non-negative integers")

    # initialize the binning arrays
    musum = numpy.zeros((Nx+2, Nmu+2))
    xsum = numpy.zeros((Nx+2, Nmu+2))
    ysum = numpy.zeros((Nell, Nx+2, Nmu+2), dtype=y3d.dtype) # extra dimension for multipoles
    Nsum = numpy.zeros((Nx+2, Nmu+2))
    
    # the line-of-sight index
    if los not in "xyz": raise ValueError("`los` must be `x`, `y`, or `z`")
    los_index = "xyz".index(los)
    
    # if input array is Hermitian symmetric, only half of the last 
    # axis is stored in `y3d`
    symmetry_axis = -1 if hermitian_symmetric else None
    
    # iterate over y-z planes of the coordinate mesh
    for slab in SlabIterator(x3d, axis=0, symmetry_axis=symmetry_axis):
        
        # the square of coordinate mesh norm 
        # (either Fourier space k or configuraton space x)
        xslab = slab.norm2()
        
        # if empty, do nothing
        if len(xslab.flat) == 0: continue

        # get the bin indices for x on the slab
        dig_x = numpy.digitize(xslab.flat, x2edges)
    
        # make xslab just x
        xslab **= 0.5
    
        # get the bin indices for mu on the slab
        mu = slab.mu(los_index) # defined with respect to specified LOS
        dig_mu = numpy.digitize(abs(mu).flat, muedges)
        
        # make the multi-index
        multi_index = numpy.ravel_multi_index([dig_x, dig_mu], (Nx+2,Nmu+2))

        # sum up x in each bin (accounting for negative freqs)
        xslab[:] *= slab.hermitian_weights
        xsum.flat += numpy.bincount(multi_index, weights=xslab.flat, minlength=xsum.size)
    
        # count number of modes in each bin (accounting for negative freqs)
        Nslab = numpy.ones_like(xslab) * slab.hermitian_weights
        Nsum.flat += numpy.bincount(multi_index, weights=Nslab.flat, minlength=Nsum.size)

        # compute multipoles by weighting by Legendre(ell, mu)
        for iell, ell in enumerate(_poles):
            
            # weight the input 3D array by the appropriate Legendre polynomial
            weighted_y3d = legpoly[iell](mu) * y3d[slab.index]

            # add conjugate for this kx, ky, kz, corresponding to 
            # the (-kx, -ky, -kz) --> need to make mu negative for conjugate
            # Below is identical to the sum of
            # Leg(ell)(+mu) * y3d[:, nonsingular]    (kx, ky, kz)
            # Leg(ell)(-mu) * y3d[:, nonsingular].conj()  (-kx, -ky, -kz)
            # or 
            # weighted_y3d[:, nonsingular] += (-1)**ell * weighted_y3d[:, nonsingular].conj()
            # but numerically more accurate.
            if hermitian_symmetric:
                
                if ell % 2: # odd, real part cancels
                    weighted_y3d.real[slab.nonsingular] = 0.
                    weighted_y3d.imag[slab.nonsingular] *= 2.
                else:  # even, imag part cancels
                    weighted_y3d.real[slab.nonsingular] *= 2.
                    weighted_y3d.imag[slab.nonsingular] = 0.
                    
            # sum up the weighted y in each bin
            weighted_y3d *= (2.*ell + 1.)
            ysum[iell,...].real.flat += numpy.bincount(multi_index, weights=weighted_y3d.real.flat, minlength=Nsum.size)
            if numpy.iscomplexobj(ysum):
                ysum[iell,...].imag.flat += numpy.bincount(multi_index, weights=weighted_y3d.imag.flat, minlength=Nsum.size)
        
        # sum up the absolute mag of mu in each bin (accounting for negative freqs)
        mu[:] *= slab.hermitian_weights
        musum.flat += numpy.bincount(multi_index, weights=abs(mu).flat, minlength=musum.size)
    
    # sum binning arrays across all ranks
    xsum  = comm.allreduce(xsum, MPI.SUM)
    musum = comm.allreduce(musum, MPI.SUM)
    ysum  = comm.allreduce(ysum, MPI.SUM)
    Nsum  = comm.allreduce(Nsum, MPI.SUM)

    # add the last 'internal' mu bin (mu == 1) to the last visible mu bin
    # this makes the last visible mu bin inclusive on both ends.
    ysum[..., -2] += ysum[..., -1]
    musum[:, -2]  += musum[:, -1]
    xsum[:, -2]   += xsum[:, -1]
    Nsum[:, -2]   += Nsum[:, -1]

    # reshape and slice to remove out of bounds points
    sl = slice(1, -1)
    with numpy.errstate(invalid='ignore'):
        
        # 2D binned results
        y2d       = (ysum[0,...] / Nsum)[sl,sl] # ell=0 is first index
        xmean_2d  = (xsum / Nsum)[sl,sl]
        mumean_2d = (musum / Nsum)[sl, sl]
        N_2d      = Nsum[sl,sl]
        
        # 1D multipole results (summing over mu (last) axis)
        if do_poles:
            N_1d     = Nsum[sl,sl].sum(axis=-1)
            xmean_1d = xsum[sl,sl].sum(axis=-1) / N_1d
            poles    = ysum[:, sl,sl].sum(axis=-1) / N_1d
            poles    = poles[ell_idx,...]
    
    # return y(x,mu) + (possibly empty) multipoles
    result = (xmean_2d, mumean_2d, y2d, N_2d)
    pole_result = (xmean_1d, poles, N_1d) if do_poles else None
    return result, pole_result
