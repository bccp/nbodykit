import numpy
from mpi4py import MPI
import logging

logger = logging.getLogger('measurestats')

def paint(field, pm):
    """
    Paint the ``DataSource`` specified by ``input`` onto the 
    ``ParticleMesh`` specified by ``pm``
    
    Parameters
    ----------
    field : ``DataSource``
        the data source object representing the field to paint onto the mesh
    pm : ``ParticleMesh``
        particle mesh object that does the painting
        
    Returns
    -------
    Ntot : int
        the total number of objects, as determined by painting
    """
    # compatibility with the older painters. 
    # We need to get rid of them.
    if hasattr(field, 'paint'):
        if pm.comm.rank == 0:
            warnings.warn('paint method of type %s shall be replaced with a read method'
                % type(field), DeprecationWarning)
        return field.paint(pm)

    pm.real[:] = 0
    Ntot = 0

    if pm.comm.rank == 0: 
        logger.info("BoxSize = %s", str(field.BoxSize))
    for position, weight in field.read(['Position', 'Weight'], pm.comm, full=False):
        min = numpy.min(
            pm.comm.allgather(
                    [numpy.inf, numpy.inf, numpy.inf] 
                    if len(position) == 0 else 
                    position.min(axis=0)),
            axis=0)
        max = numpy.max(
            pm.comm.allgather(
                    [-numpy.inf, -numpy.inf, -numpy.inf] 
                    if len(position) == 0 else 
                    position.max(axis=0)),
            axis=0)
        if pm.comm.rank == 0:
            logger.info("Range of position %s:%s" % (str(min), str(max)))

        layout = pm.decompose(position)
        # Ntot shall be calculated before exchange. Issue #55.
        if weight is None:
            Ntot += len(position)
            weight = 1
        else:
            Ntot += weight.sum()
            weight = layout.exchange(weight)
        position = layout.exchange(position)

        pm.paint(position, weight)
    return pm.comm.allreduce(Ntot)


def compute_3d_power(fields, pm, transfer=[], painter=paint, comm=None, log_level=logging.DEBUG):
    """
    Compute and return the 3D power from two input fields
    
    Parameters
    ----------
    fields : list of ``DataSource`` objects
        the list of fields from which the 3D power will be computed
        
    pm : ``ParticleMesh``
        particle mesh object that does the painting
        
    transfer : list, optional
        A chain of transfer functions to apply to the complex field.
        
    painter : callable, optional
        The function used to 'paint' the fields onto the particle mesh.
        Default is ``measurestats.paint`` -- see documentation for 
        required API of user-supplied functions
        
    comm : MPI.Communicator, optional
        the communicator to pass to the ``ParticleMesh`` object. If not
        provided, ``MPI.COMM_WORLD`` is used
        
    log_level : int, optional
        logging level to use while computing. Default is ``logging.DEBUG``
        which has numeric value of `10`
    
    Returns
    -------
    p3d : array_like (real)
        the 3D power spectrum, corresponding to the gridded input fields
        
    N1 : int
        the total number of objects in the 1st field
        
    N2 : int
        the total number of objects in the 2nd field (equal to N1 if 
        only one input field specified)
    """
    # some setup
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    if log_level is not None: logger.setLevel(log_level)
        
    # paint, FT field and filter field #1
    N1 = painter(fields[0], pm)
    if rank == 0: logger.info('painting done')
    pm.r2c()
    if rank == 0: logger.info('r2c done')
    pm.transfer(transfer)

    # do the cross power
    do_cross = len(fields) > 1 and fields[0] != fields[1]
    if do_cross:
                
        # crash if box size isn't the same
        if not numpy.all(fields[0].BoxSize == fields[1].BoxSize):
            raise ValueError("mismatch in box sizes for cross power measurement")
        
        # copy and store field #1's complex
        c1 = pm.complex.copy()
        
        # paint, FT, and filter field #2
        N2 = painter(fields[1], pm)
        if rank == 0: logger.info('painting 2 done')
        pm.r2c()
        if rank == 0: logger.info('r2c 2 done')
        pm.transfer(transfer)
        c2 = pm.complex
  
    # do the auto power
    else:
        c1 = pm.complex
        c2 = pm.complex
        N2 = N1

    # reuse the memory in c1.real for the 3d power spectrum
    p3d = c1.real

    # calculate the 3d power spectrum, row by row to save memory
    for row in range(len(c1)):
        p3d[row, ...] = c1[row].real * c2[row].real + c1[row].imag * c2[row].imag

    # the complex field is dimensionless; power is L^3
    # ref to http://icc.dur.ac.uk/~tt/Lectures/UA/L4/cosmology.pdf
    p3d[...] *= pm.BoxSize.prod() 
                
    return p3d, N1, N2


def compute_brutal_corr(fields, Rmax, Nr, Nmu=0, comm=None, subsample=1, los='z', poles=[]):
    """
    Compute the correlation function by direct pair summation, projected
    into either 1d `R` bins or 2d (`R`, `mu`) bins
    
    Parameters
    ----------
    fields : list of ``DataSource`` objects
        the list of fields from which the 3D correlation will be computed
        
    Rmax : float
        the maximum R value to compute, in the same units as the input
        fields
    
    Nmu : int, optional
        the number of desired `mu` bins, where `mu` is the cosine 
        of the angle from the line-of-sight. Default is 0, in 
        which case the correlation function is binned as a function of 
        `R` only
        
    comm : MPI.Communicator, optional
        the communicator to pass to the ``ParticleMesh`` object. If not
        provided, ``MPI.COMM_WORLD`` is used
        
    subsample : int, optional
        Down-sample the input fields by choosing 1 out of every N points. 
        Default is `1`
    
    los : 'x', 'y', 'z', optional
        the dimension to treat as the line-of-sight; default is 'z'
        
    Returns
    -------
    pc : ``correlate.paircount``
        the ``kdcount`` pair counting instance
    xi : array_like
        the correlation function
    RR : array_like
        the number of random-random pairs (used as normalization of the data-data pairs)
    """
    from pmesh.domain import GridND
    from kdcount import correlate
    
    if los not in "xyz":
        raise ValueError("the `los` must be one of `x`, `y`, or `z`")
    los = "xyz".index(los)
    poles = numpy.array(poles)
    
    # the comm
    if comm is None: comm = MPI.COMM_WORLD
    
    # determine processors for grididng
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
    if comm.rank == 0:
        logger.info('Nproc = %s' %str(Nproc))
        logger.info('Rmax = %g' %Rmax)
    
    # domain decomposition
    grid = [numpy.linspace(0, fields[0].BoxSize[i], Nproc[i]+1, endpoint=True) for i in range(3)]
    domain = GridND(grid, comm=comm)

    # read position for field #1 
    [[pos1]] = fields[0].read(['Position'], comm, full=False)
    pos1 = pos1[comm.rank * subsample // comm.size ::subsample]
    N1 = comm.allreduce(len(pos1))
    
    # read position for field #2
    if len(fields) > 1:
        [[pos2]] = fields[1].read(['Position'], comm, full=False)
        pos2 = pos2[comm.rank * subsample // comm.size ::subsample]
        N2 = comm.allreduce(len(pos2))
    else:
        pos2 = pos1
        N2 = N1
    
    # exchange field #1 positions    
    layout = domain.decompose(pos1, smoothing=0)
    pos1 = layout.exchange(pos1)
    if comm.rank == 0:
        logger.info('exchange pos1')
        
    # exchange field #2 positions
    if Rmax > fields[0].BoxSize[0] * 0.25:
        pos2 = numpy.concatenate(comm.allgather(pos2), axis=0)
    else:
        layout = domain.decompose(pos2, smoothing=Rmax)
        pos2 = layout.exchange(pos2)
    if comm.rank == 0:
        logger.info('exchange pos2')

    # initialize the points trees
    tree1 = correlate.points(pos1, boxsize=fields[0].BoxSize)
    tree2 = correlate.points(pos2, boxsize=fields[0].BoxSize)

    logger.info('rank %d correlating %d x %d' %(comm.rank, len(tree1), len(tree2)))
    if comm.rank == 0:
        logger.info('all correlating %d x %d' %(N1, N2))

    # the binning, either r or (r,mu)
    if len(poles):
        bins = correlate.FlatSkyMultipoleBinning(Rmax, Nr, poles, los, compute_mean_coords=True)
    elif Nmu > 0:
        bins = correlate.FlatSkyBinning(Rmax, Nr, Nmu, los, compute_mean_coords=True)
    else:
        bins = correlate.RBinning(Rmax, Nr, compute_mean_coords=True)

    # do the pair count
    # have to set usefast = False to get mean centers, or exception thrown
    pc = correlate.paircount(tree2, tree1, bins, np=0, usefast=False)
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
    RR = 1.*N1*N2 / fields[0].BoxSize.prod()
    if Nmu > 0:
        dr3 = numpy.diff(pc.edges[0]**3)
        dmu = numpy.diff(pc.edges[1])
        RR *= 2. / 3. * numpy.pi * dr3[:,None] * dmu[None,:]
    else:
        RR *= 4. / 3. * numpy.pi * numpy.diff(pc.edges**3)
    
    # return the correlation and the pair count object
    xi = (1. * pc.sum1 / RR) - 1.0
    if len(poles):
        xi = xi.T # make ell the second axis 
        xi[:,poles!=0] += 1.0 # only monopole gets minus one

    return pc, xi, RR

def compute_3d_corr(fields, pm, transfer=[], painter=paint, comm=None, log_level=logging.DEBUG):
    """
    Compute the 3d correlation function by Fourier transforming 
    the 3d power spectrum
    
    See documentation of `measurestats.compute_3d_power`
    """
    kwargs = {'transfer':transfer, 'painter':painter, 'comm':comm, 'log_level':log_level}
    p3d, N1, N2 = compute_3d_power(fields, pm, **kwargs)
    
    # directly transform dimensionless p3d
    # Note that L^3 cancels with dk^3.
    pm.complex[:] = p3d.copy()
    pm.c2r()
    xi3d = pm.real
    
    return xi3d, N1, N2


def project_to_basis(comm, x3d, y3d, edges, los='z', poles=[], symmetric=True):
    """ 
    Project a 3D statistic on to the specified basis. 

    The projection will be onto 2d bins `(x, mu)`, where `x`
    is separation `r` in configuration space or wavenumber `k` in 
    Fourier space, and `mu` is the cosine of the angle to the 
    line-of-sight. 
    
    Optionally, the multipoles of the 2d `(x, mu)` bins are 
    also returned, as specified by the multipole numbers in `poles`
    
    Notes
    -----
    *   the mu range extends from 0 to 1.0
    *   the mu bins are half-inclusive half-exclusive, except the last bin
        is inclusive on both ends (to include mu = 1.0)
    *   when Nmu == 1, the case reduces to the isotropic 1D binning
    
    Parameters
    ----------
    comm : MPI.Comm
        the communicator for the decomposition of the power spectrum
        
    x3d  : list
        The list of x values for each item in the `y3d` array. The items
        must broadcast to the same shape of `y3d`.

    y3d : array_like (real)
        a 3D statistic, either a power spectrum (defined in Fourier space),
        or a correlation function (defined in configuration space)

    edges : array_like
        an array specifying the edges of the `x` bins, where `x` is 
        either Fourier space `k` or configuration space `r`

    Nmu : int
        the number of mu bins to use when binning in the 3d statistic
    
    los : str, {'x','y','z'}
        the line-of-sight direction, which the angle `mu` is defined with
        respect to. Default is `z`.
        
    poles : list of int, optional
        if provided, a list of integers specifying multipole numbers to
        project the 2d `(x, mu)` on to
        
    symmetric : bool, optional
        If `True`, the `y3d` area is assumed to be symmetric about the `z = 0`
        plane. If `y3d` is a power spectrum, this should be set to `True`, 
        while if `y3d` is a correlation function, this should be `False`
        
    Returns
    -------
    edges : list 
        a list of [x_edges, ]
    """
    from scipy.special import legendre
        
    # bin edges
    xedges, muedges = edges
    Nx = len(xedges) - 1 
    Nmu = len(muedges) - 1
    
    # always measure make sure first ell is monopole, which
    # is just (x, mu) projection since legendre of ell=0 is 1
    do_poles = len(poles) > 0
    poles_ = [0]+sorted(poles) if 0 not in poles else sorted(poles)
    ell_idx = [poles_.index(l) for l in poles]
    Nell = len(poles_)
    
    # valid ell values
    if any(ell < 0 for ell in poles_):
        raise RuntimeError("multipole numbers must be nonnegative integers")

    # squared x bin edges
    xedges2 = xedges ** 2

    musum = numpy.zeros((Nx+2, Nmu+2))
    xsum = numpy.zeros((Nx+2, Nmu+2))
    ysum = numpy.zeros((Nell, Nx+2, Nmu+2)) # extra dimension for multipoles
    Nsum = numpy.zeros((Nx+2, Nmu+2))
    
    # los index
    los_index = 'xyz'.index(los)
    
    # count everything but z = 0 plane twice (r2c transform stores 1/2 modes)
    nonsingular = numpy.squeeze(x3d[2] != 0) # has length of Nz now
    
    for row in range(len(x3d[0])):
        
        # now scratch stores x3d ** 2
        scratch = numpy.float64(x3d[0][row] ** 2)
        for xi in x3d[1:]:
            scratch = scratch + xi[0] ** 2

        if len(scratch.flat) == 0:
            # no data
            continue

        dig_x = numpy.digitize(scratch.flat, xedges2)
    
        # make scratch just x
        scratch **= 0.5
    
        # store mu
        with numpy.errstate(invalid='ignore'):
            if los_index == 0:
                mu = abs(x3d[los_index][row]/scratch)
            else:
                mu = abs(x3d[los_index][0]/scratch)
        dig_mu = numpy.digitize(mu.flat, muedges)
        
        # make the multi-index
        multi_index = numpy.ravel_multi_index([dig_x, dig_mu], (Nx+2,Nmu+2))
    
        # count modes not in singular plane twice
        if symmetric: scratch[:, nonsingular] *= 2.
    
        # the x sum
        xsum.flat += numpy.bincount(multi_index, weights=scratch.flat, minlength=xsum.size)
    
        # take the sum of weights
        scratch[...] = 1.0
        # count modes not in singular plane twice
        if symmetric: scratch[:, nonsingular] = 2.
        Nsum.flat += numpy.bincount(multi_index, weights=scratch.flat, minlength=Nsum.size)

        scratch[...] = y3d[row]

        # the singular plane is down weighted by 0.5
        if symmetric: scratch[:, nonsingular] *= 2.
        
        # weight P(k,mu) and sum the weighted values
        for iell, ell in enumerate(poles_):
            weighted_y3d = scratch * (2*ell + 1.) * legendre(ell)(mu)
            ysum[iell,...].flat += numpy.bincount(multi_index, weights=weighted_y3d.flat, minlength=Nsum.size)
        
        # the mu sum
        if symmetric: mu[:, nonsingular] *= 2.
        musum.flat += numpy.bincount(multi_index, weights=mu.flat, minlength=musum.size)

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
        
        # projected results
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
    
    # return just y(x,mu) or y(x,mu) + multipoles
    result = (xmean_2d, mumean_2d, y2d, N_2d)
    pole_result = (xmean_1d, poles, N_1d) if do_poles else None
    
    return result, pole_result
