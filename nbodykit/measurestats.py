import numpy
from mpi4py import MPI
import logging
from nbodykit.extensionpoints import Painter, Transfer
import time

logger = logging.getLogger('measurestats')

def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def compute_3d_power(fields, pm, comm=None):
    """
    Compute and return the 3D power from two input fields
    
    Parameters
    ----------
    fields : list of (``DataSource``, ``Painter``, ``Transfer``) tuples
        the list of fields which the 3D power will be computed
        
    pm : ``ParticleMesh``
        particle mesh object that handles the painting and FFTs
                
    comm : MPI.Communicator, optional
        the communicator to pass to the ``ParticleMesh`` object. If not
        provided, ``MPI.COMM_WORLD`` is used
        
    Returns
    -------
    p3d : array_like (real)
        the 3D power spectrum, corresponding to the gridded input fields
        
    stats1 : dict
        dictionary holding the statistics of the first field, as returned
        by the `Painter` used 
        
    stats2 : dict
        dictionary holding the statistics of the second field, as returned
        by the `Painter` used
    """
    # some setup
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    
    # check that the painter was passed correctly
    datasources = [d for d, p, t in fields]
    painters    = [p for d, p, t in fields]
    transfers   = [t for d, p, t in fields]
    
    # paint, FT field and filter field #1
    stats1 = painters[0].paint(pm, datasources[0])
    if rank == 0: logger.info('%s painting done' %pm.paintbrush)
    pm.r2c()
    if rank == 0: logger.info('r2c done')
    pm.transfer(transfers[0])

    # do the cross power if two fields supplied
    if len(fields) > 1:
                
        # crash if box size isn't the same
        if not numpy.all(datasources[0].BoxSize == datasources[1].BoxSize):
            raise ValueError("mismatch in box sizes for cross power measurement")
        
        # copy and store field #1's complex
        c1 = pm.complex.copy()
        
        # paint, FT, and filter field #2
        stats2 = painters[1].paint(pm, datasources[1])
        if rank == 0: logger.info('%s painting 2 done' %pm.paintbrush)
        pm.r2c()
        if rank == 0: logger.info('r2c 2 done')
        pm.transfer(transfers[1])
        c2 = pm.complex
  
    # do the auto power
    else:
        c1 = pm.complex
        c2 = pm.complex
        stats2 = stats1

    # reuse the memory in c1.real for the 3d power spectrum
    p3d = c1
    
    # calculate the 3d power spectrum, islab by islab to save memory
    for islab in range(len(c1)):
        p3d[islab, ...] = c1[islab]*c2[islab].conj()

    # the complex field is dimensionless; power is L^3
    # ref to http://icc.dur.ac.uk/~tt/Lectures/UA/L4/cosmology.pdf
    p3d[...] *= pm.BoxSize.prod() 
                
    return p3d, stats1, stats2
    
def compute_bianchi_poles(comm, max_ell, catalog, Nmesh, 
                            factor_hexadecapole=False, 
                            paintbrush='cic'):
    """
    Compute and return the 3D power multipoles (ell = [0, 2, 4]) from one 
    input field, which contains non-trivial survey geometry.
    
    The estimator uses the FFT algorithm outlined in Bianchi et al. 2015 to compute
    the monopole, quadrupole, and hexadecapole
    
    Parameters
    ----------
    max_ell : int
        the maximum multipole number to compute up to (and including). Must be
        one of [0, 2, 4]
        
    datasource : ``DataSource``
        the DataSource which will be return the Cartesian coordinates of the
        (ra, dec, z) survey
        
    pm : ``ParticleMesh``
        particle mesh object that handles the painting and FFTs
            
    factor_hexadecapole: bool, optional
        if `True`, use the factored expression for the hexadecapole (ell=4) from
        eq. 27 of Scoccimarro 2015 (1506.02729); default is `False`
    
    Returns
    -------
    poles : list
        list of the 3D power multipoles computed, up to and including `ell=max_ell`
        
    stats : dict
        dictionary holding the statistics of the input datasource, as returned
        by the `FKPPainter` painter
    """
    from pmesh.particlemesh import ParticleMesh
    
    def bianchi_transfer(data, x, i, j, k=None, offset=[0., 0., 0.]):
        """
        Transfer functions necessary to compute the power spectrum  
        multipoles via FFTs
        
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
        offset : array_like
            add an average offset to the `x` component arrays; 
            needed if the simulation box has an average offset 
            from the grid box in configuration-space
        """
        # compute x**2
        norm = sum((xi + offset[ii])**2 for ii, xi in enumerate(x))
    
        # get x_i, x_j
        # if i == 'x' direction, it's just one value
        xi = x[i] + offset[i]
        xj = x[j] + offset[j]

        # multiply the kernel
        if k is not None:
            xk = x[k] + offset[k]
            data[:] *= xi**2 * xj * xk
            idx = norm != 0.
            data[idx] /= norm[idx]**2
        
        else:
            data[:] *= xi * xj
            idx = norm != 0.
            data[idx] /= norm[idx]
    
    # the rank
    rank = comm.rank
    
    # the painting kernel transfer
    if paintbrush == 'cic':
        transfer = Transfer.create('AnisotropicCIC')
    elif paintbrush == 'tsc':
        transfer = Transfer.create('AnisotropicTSC')
    else:
        raise ValueError("valid `paintbrush` values are: ['cic', 'tsc']")

    # which transfers do we need
    if max_ell not in [0, 2, 4]:
        raise ValueError("valid values for the maximum multipole number are [0, 2, 4]")
    ells = numpy.arange(0, max_ell+1, 2)
    bianchi_transfers = []
    
    # quadrupole kernels
    if max_ell > 0:
        k2 = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        a2 = [1.]*3 + [2.]*3
        bianchi_transfers.append((a2, k2))
    
    # hexadecapole kernels
    if max_ell > 2 and not factor_hexadecapole:
        k4 = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2),
             (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 1), (0, 1, 1),
             (0, 2, 2), (1, 2, 2), (0, 1, 2), (1, 0, 2), (2, 0, 1)]
        a4 = [1.]*3 + [4.]*6 + [6.]*3 + [12.]*3
        bianchi_transfers.append((a4, k4))
    
    # load the data/randoms and setup boxsize, etc
    with catalog:
        
        # the mean coordinate offset
        offset = catalog.mean_coordinate_offset
        
        # initialize the particle mesh
        pm = ParticleMesh(catalog.BoxSize, Nmesh, paintbrush=paintbrush, dtype='f8', comm=comm)
        
        # do the FKP painting
        stats = catalog.paint(pm)
    
    # save the fkp density for later
    density = pm.real.copy()
    if rank == 0: logger.info('%s painting done' %paintbrush)
    
    # compute the monopole, A0(k), and save
    pm.r2c()
    transfer(pm, pm.complex)
    A0 = pm.complex.copy()
    if rank == 0: logger.info('ell = 0 done; 1 r2c completed')
    
    volume = pm.BoxSize.prod()
    
    # store the A0, A2, A4 arrays
    result = []
    result.append(A0*volume)
    
    # the x grid points (at point centers)
    cell_size = pm.BoxSize / pm.Nmesh
    xgrid = [(ri+0.5)*cell_size[i] for i, ri in enumerate(pm.r)]
    
    start = time.time()
    for iell in range(len(bianchi_transfers)):
        ell = ells[iell+1]
        A_ell = numpy.zeros_like(pm.complex)
        
        for amp, integers in zip(*bianchi_transfers[iell]):
                        
            # reset the 'pm.real' array to the original FKP density
            pm.real[:] = density[:]
        
            # apply the real-space transfer
            if rank == 0: logger.debug("applying real-space Bianchi transfer for %s..." %str(integers))
            bianchi_transfer(pm.real, xgrid, *integers, offset=offset)
            if rank == 0: logger.debug('...done')
    
            # do the FT and apply the k-space kernel
            if rank == 0: logger.debug("performing r2c...")
            pm.r2c()
            if rank == 0: logger.debug('...done')
            
            # fourier space kernel
            if rank == 0: logger.debug("applying Fourier-space Bianchi transfer for %s..." %str(integers))
            bianchi_transfer(pm.complex, pm.k, *integers)
            if rank == 0: logger.debug('...done')
            
            # and save
            A_ell[:] += amp*pm.complex[:]*volume
            
        # apply the gridding transfer and save
        transfer(pm, A_ell)
        result.append(A_ell)
        if rank == 0: 
            args = (ell, len(bianchi_transfers[iell][0]))
            logger.info('ell = %d done; %s r2c completed' %args)
        
    stop = time.time()
    if rank == 0:
        logger.info("higher order multipoles computed in elapsed time %s" %timer(start, stop))
        if factor_hexadecapole:
            logger.info("using factorized hexadecapole estimator for ell=4")
    
    # proper normalization: A_ran from equation 
    norm = 1.0 / stats['A_ran']
    
    # reuse memory for output
    P0 = result[0]
    if max_ell > 0: P2 = result[1]
    if max_ell > 2:
        if not factor_hexadecapole: 
            P4 = result[2]
        else:
            P4 = numpy.empty_like(P2)
        
    # calculate the multipoles, islab by islab to save memory
    # see equations 6-8 of Bianchi et al. 2015
    for islab in range(len(P0)):

        # save for reuse
        P0_star = (P0[islab]).conj()
        if max_ell > 0: P2_star = (P2[islab]).conj()

        # hexadecapole
        if max_ell > 2:
            if not factor_hexadecapole:
                P4[islab, ...] = norm * 9./8. * P0[islab] * (35.*(P4[islab]).conj() - 30.*P2_star + 3.*P0_star)
            else:
                P4[islab, ...] = norm * 9./8. * ( 35.*P2[islab]*P2_star + 3.*P0[islab]*P0_star - 5./3.*(11.*P0[islab]*P2_star + 7.*P2[islab]*P0_star) )
        # quadrupole
        if max_ell > 0:
            P2[islab, ...] = norm * 5./2. * P0[islab] * (3.*P2_star - P0_star)

        # monopole
        P0[islab, ...] = norm * P0[islab] * P0_star
        
    result = [P0]
    if max_ell > 0: result.append(P2)
    if max_ell > 2: result.append(P4)

    return pm, result, stats


def compute_brutal_corr(datasources, rbins, Nmu=0, comm=None, subsample=1, los='z', poles=[]):
    """
    Compute the correlation function by direct pair summation, projected
    into either 1d `R` bins or 2d (`R`, `mu`) bins
    
    Parameters
    ----------
    datasources : list of ``DataSource`` objects
        the list of datasources from which the 3D correlation will be computed
        
    Rmax : float
        the maximum R value to compute, in the same units as the input
        datasources
    
    Nmu : int, optional
        the number of desired `mu` bins, where `mu` is the cosine 
        of the angle from the line-of-sight. Default is 0, in 
        which case the correlation function is binned as a function of 
        `R` only
        
    comm : MPI.Communicator, optional
        the communicator to pass to the ``ParticleMesh`` object. If not
        provided, ``MPI.COMM_WORLD`` is used
        
    subsample : int, optional
        Down-sample the input datasources by choosing 1 out of every N points. 
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
    Rmax = rbins[-1]
    
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
    if comm.rank == 0:
        logger.info('exchange pos1')
        
    # exchange field #2 positions
    if Rmax > datasources[0].BoxSize[0] * 0.25:
        pos2 = numpy.concatenate(comm.allgather(pos2), axis=0)
    else:
        layout = domain.decompose(pos2, smoothing=Rmax)
        pos2 = layout.exchange(pos2)
    if comm.rank == 0:
        logger.info('exchange pos2')

    # initialize the points trees
    tree1 = correlate.points(pos1, boxsize=datasources[0].BoxSize)
    tree2 = correlate.points(pos2, boxsize=datasources[0].BoxSize)

    logger.info('rank %d correlating %d x %d' %(comm.rank, len(tree1), len(tree2)))
    if comm.rank == 0:
        logger.info('all correlating %d x %d' %(N1, N2))

    # the binning, either r or (r,mu)
    if len(poles):
        bins = correlate.FlatSkyMultipoleBinning(rbins, poles, los, compute_mean_coords=True)
    elif Nmu > 0:
        bins = correlate.FlatSkyBinning(rbins, Nmu, los, compute_mean_coords=True)
    else:
        bins = correlate.RBinning(rbins, compute_mean_coords=True)

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
        xi = xi.T # make ell the second axis 
        xi[:,poles!=0] += 1.0 # only monopole gets minus one

    return pc, xi, RR

def compute_3d_corr(fields, pm, comm=None):
    """
    Compute the 3d correlation function by Fourier transforming 
    the 3d power spectrum
    
    See documentation of `measurestats.compute_3d_power`
    """

    p3d, N1, N2 = compute_3d_power(fields, pm, comm=comm)
    
    # directly transform dimensionless p3d
    # Note that L^3 cancels with dk^3.
    pm.complex[:] = p3d.copy()
    pm.complex[:] *= 1.0 / pm.BoxSize.prod()
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
    legpoly = [legendre(l) for l in poles_]
    ell_idx = [poles_.index(l) for l in poles]
    Nell = len(poles_)
    
    # valid ell values
    if any(ell < 0 for ell in poles_):
        raise RuntimeError("multipole numbers must be nonnegative integers")

    # squared x bin edges
    xedges2 = xedges ** 2

    musum = numpy.zeros((Nx+2, Nmu+2))
    xsum = numpy.zeros((Nx+2, Nmu+2))
    ysum = numpy.zeros((Nell, Nx+2, Nmu+2), dtype=y3d.dtype) # extra dimension for multipoles
    Nsum = numpy.zeros((Nx+2, Nmu+2))
    
    # los index
    los_index = 'xyz'.index(los)
    
    # need to count all modes with positive z frequency twice due to r2c FFTs
    nonsingular = numpy.squeeze(x3d[2] > 0.) # has length of Nz now

    for islab in range(len(x3d[0])):
        
        # now xslab stores x3d ** 2
        xslab = numpy.float64(x3d[0][islab] ** 2)
        for xi in x3d[1:]:
            xslab = xslab + xi[0] ** 2

        if len(xslab.flat) == 0:
            # no data
            continue

        dig_x = numpy.digitize(xslab.flat, xedges2)
    
        # make xslab just x
        xslab **= 0.5
    
        # store mu (keeping track of positive/negative)
        with numpy.errstate(invalid='ignore'):
            if los_index == 0:
                mu = x3d[los_index][islab]/xslab
            else:
                mu = x3d[los_index][0]/xslab
        dig_mu = numpy.digitize(abs(mu).flat, muedges)
        
        # make the multi-index
        multi_index = numpy.ravel_multi_index([dig_x, dig_mu], (Nx+2,Nmu+2))
    
        # count modes not in singular plane twice
        if symmetric: xslab[:, nonsingular] *= 2.
    
        # the x sum
        xsum.flat += numpy.bincount(multi_index, weights=xslab.flat, minlength=xsum.size)
    
        # count number of modes
        Nslab = numpy.ones_like(xslab)
        if symmetric: Nslab[:, nonsingular] = 2. # count modes not in singular plane twice
        Nsum.flat += numpy.bincount(multi_index, weights=Nslab.flat, minlength=Nsum.size)

        # weight P(k,mu) and sum for the poles
        for iell, ell in enumerate(poles_):
            
            weighted_y3d = legpoly[iell](mu) * y3d[islab]

            # add conjugate for this kx, ky, kz, corresponding to 
            # the (-kx, -ky, -kz) --> need to make mu negative for conjugate
            # Below is identical to the sum of
            # Leg(ell)(+mu) * y3d[:, nonsingular]    (kx, ky, kz)
            # Leg(ell)(-mu) * y3d[:, nonsingular].conj()  (-kx, -ky, -kz)
            # or 
            # weighted_y3d[:, nonsingular] += (-1)**ell * weighted_y3d[:, nonsingular].conj()
            # but numerically more accurate.
            if symmetric:
                if ell % 2: # odd, real part cancels
                    weighted_y3d.real[:, nonsingular] = 0.
                    weighted_y3d.imag[:, nonsingular] *= 2.
                else:  # even, imag part cancels
                    weighted_y3d.real[:, nonsingular] *= 2.
                    weighted_y3d.imag[:, nonsingular] = 0.
                    
            weighted_y3d *= (2.*ell + 1.)
            ysum[iell,...].real.flat += numpy.bincount(multi_index, weights=weighted_y3d.real.flat, minlength=Nsum.size)
            if numpy.iscomplexobj(ysum):
                ysum[iell,...].imag.flat += numpy.bincount(multi_index, weights=weighted_y3d.imag.flat, minlength=Nsum.size)
        
        # the mu sum
        if symmetric: mu[:, nonsingular] *= 2.
        musum.flat += numpy.bincount(multi_index, weights=abs(mu).flat, minlength=musum.size)

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
