from mpi4py import MPI
import numpy
import logging
from nbodykit.meshtools import SlabIterator

logger = logging.getLogger('measurestats')

#------------------------------------------------------------------------------
# Fourier space statistics
#------------------------------------------------------------------------------
def compute_3d_power(fields, pm, transfers=[], comm=None):
    """
    Compute and return the 3D power from two input fields

    Parameters
    ----------
    fields : list of sources
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

    # make sure we have right number of transfer functions
    if len(fields) == 2 and len(transfers) == 1:
        transfers = [transfers]*2

    # step 1: paint the density field to the mesh
    real = fields[0].paint(pm)
    if rank == 0: logger.info('field #1: %s painting done' %fields[0].painter.paintbrush)

    # step 2: Fourier transform density field using real to complex FFT
    c1 = real.r2c()
    del real
    if rank == 0: logger.info('field #1: r2c done')

    # step 3: apply transfer function kernels to complex field
    for fk in transfers[0]: fk(c1)

    # compute the auto power of single supplied field
    if len(fields) == 1:
        c2 = c1

    # compute the cross power of the two supplied fields
    else:        
        # apply painting, FFT, and transfer steps to second field
        real = fields[1].paint(pm)
        if rank == 0: logger.info('field #2: %s painting done' %fields[1].painter.paintbrush)
        
        # FFT
        c2 = real.r2c()
        del real
        if rank == 0: logger.info('field #2: r2c done')

        # transfers
        for fk in transfers[1]: fk(c2)

    # calculate the 3d power spectrum, slab-by-slab to save memory
    p3d = c1
    for (s0, s1, s2) in zip(p3d.slabs, c1.slabs, c2.slabs):
        s0[...] = s1 * s2.conj()

    # the complex field is dimensionless; power is L^3
    # ref to http://icc.dur.ac.uk/~tt/Lectures/UA/L4/cosmology.pdf
    p3d[...] *= pm.BoxSize.prod() 

    return p3d

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
    Nsum = numpy.zeros((Nx+2, Nmu+2), dtype='i8')
    
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