import numpy
import numbers
from mpi4py import MPI

from pmesh.pm import RealField, ComplexField
from nbodykit.meshtools import SlabIterator
from nbodykit.utils import GatherArray, ScatterArray
from nbodykit.mpirng import MPIRandomState
import mpsort

def gaussian_complex_fields(pm, linear_power, seed,
            unitary_amplitude=False, inverted_phase=False,
            compute_displacement=False, logger=None):
    r"""
    Make a Gaussian realization of a overdensity field, :math:`\delta(x)`.

    If specified, also compute the corresponding 1st order Lagrangian
    displacement field (Zel'dovich approximation) :math:`\psi(x)`,
    which is related to the linear velocity field via:

    .. math::

        v(x) = f a H \psi(x)

    Notes
    -----
    This computes the overdensity field using the following steps:

    #. Generate complex variates with unity variance
    #. Scale the Fourier field by :math:`(P(k) / V)^{1/2}`

    After step 2, the complex field has unity variance. This
    is equivalent to generating real-space normal variates
    with mean and unity variance, calling r2c() and dividing by :math:`N^3`
    since the variance of the complex FFT (with no additional normalization)
    is :math:`N^3 \times \sigma^2_\mathrm{real}`.

    Furthermore, the power spectrum is defined as V * variance.
    So a normalization factor of 1 / V shows up in step 2,
    cancels this factor such that the power spectrum is P(k).

    The linear displacement field is computed as:

    .. math::

        \psi_i(k) = i \frac{k_i}{k^2} \delta(k)

    .. note::

        To recover the linear velocity in proper units, i.e., km/s,
        from the linear displacement, an additional factor of
        :math:`f \times a \times H(a)` is required

    Parameters
    ----------
    pm : pmesh.pm.ParticleMesh
        the mesh object
    linear_power : callable
        a function taking wavenumber as its only argument, which returns
        the linear power spectrum
    seed : int
        the random seed used to generate the random field
    compute_displacement : bool, optional
        if ``True``, also return the linear Zel'dovich displacement field;
        default is ``False``
    unitary_amplitude : bool, optional
        if ``True``, the seed gaussian has unitary_amplitude.
    inverted_phase: bool, optional
        if ``True``, the phase of the seed gaussian is inverted

    Returns
    -------
    delta_k : ComplexField
        the real-space Gaussian overdensity field
    disp_k : ComplexField or ``None``
        if requested, the Gaussian displacement field
    """
    if not isinstance(seed, numbers.Integral):
        raise ValueError("the seed used to generate the linear field must be an integer")

    if logger and pm.comm.rank == 0:
        logger.info("Generating whitenoise")

    # use pmesh to generate random complex white noise field (done in parallel)
    # variance of complex field is unity
    # multiply by P(k)**0.5 to get desired variance
    delta_k = pm.generate_whitenoise(seed, type='untransposedcomplex', unitary=unitary_amplitude)

    if logger and pm.comm.rank == 0:
        logger.info("Write noise generated")

    if inverted_phase: delta_k[...] *= -1

    # initialize the displacement fields for (x,y,z)
    if compute_displacement:
        disp_k = [pm.create(type='untransposedcomplex') for i in range(delta_k.ndim)]
        for i in range(delta_k.ndim): disp_k[i][:] = 1j
    else:
        disp_k = None

    # volume factor needed for normalization
    norm = 1.0 / pm.BoxSize.prod()

    # iterate in slabs over fields
    slabs = [delta_k.slabs.x, delta_k.slabs]
    if compute_displacement:
        slabs += [d.slabs for d in disp_k]

    # loop over the mesh, slab by slab
    for islabs in zip(*slabs):
        kslab, delta_slab = islabs[:2] # the k arrays and delta slab

        # the square of the norm of k on the mesh
        k2 = sum(kk**2 for kk in kslab)
        zero_idx = k2 == 0.

        k2[zero_idx] = 1. # avoid dividing by zero

        # the linear power (function of k)
        power = linear_power((k2**0.5).flatten())

        # multiply complex field by sqrt of power
        delta_slab[...].flat *= (power*norm)**0.5

        # set k == 0 to zero (zero config-space mean)
        delta_slab[zero_idx] = 0.

        # compute the displacement
        if compute_displacement:

            # ignore division where k==0 and set to 0
            with numpy.errstate(invalid='ignore', divide='ignore'):
                for i in range(delta_k.ndim):
                    disp_slab = islabs[2+i]
                    disp_slab[...] *= kslab[i] / k2 * delta_slab[...]
                    disp_slab[zero_idx] = 0. # no bulk displacement

    if logger and pm.comm.rank == 0:
        logger.info("Displacement computed in fourier space")

    # return Fourier-space density and displacement (which could be None)
    return delta_k, disp_k


def gaussian_real_fields(pm, linear_power, seed,
                unitary_amplitude=False,
                inverted_phase=False, compute_displacement=False, logger=None):
    r"""
    Make a Gaussian realization of a overdensity field in
    real-space :math:`\delta(x)`.

    If specified, also compute the corresponding linear Zel'dovich
    displacement field :math:`\psi(x)`, which is related to the
    linear velocity field via:

    Notes
    -----
    See the docstring for :func:`gaussian_complex_fields` for the
    steps involved in generating the fields.

    Parameters
    ----------
    pm : pmesh.pm.ParticleMesh
        the mesh object
    linear_power : callable
        a function taking wavenumber as its only argument, which returns
        the linear power spectrum
    seed : int
        the random seed used to generate the random field
    compute_displacement : bool, optional
        if ``True``, also return the linear Zel'dovich displacement field;
        default is ``False``
    unitary_amplitude : bool, optional
        if ``True``, the seed gaussian has unitary_amplitude.
    inverted_phase: bool, optional
        if ``True``, the phase of the seed gaussian is inverted

    Returns
    -------
    delta : RealField
        the real-space Gaussian overdensity field
    disp : RealField or ``None``
        if requested, the Gaussian displacement field
    """
    # make fourier fields
    delta_k, disp_k = gaussian_complex_fields(pm, linear_power, seed,
                            inverted_phase=inverted_phase,
                            unitary_amplitude=unitary_amplitude,
                            compute_displacement=compute_displacement,
                            logger=logger)

    # FFT the density to real-space
    delta = delta_k.c2r()

    std = (delta ** 2).cmean() ** 0.5

    if logger and pm.comm.rank == 0:
        logger.info("Overdensity computed in configuration space: std = %s" % str(std))

    # FFT the velocity back to real space
    if compute_displacement:
        disp = [disp_k[i].c2r() for i in range(delta.ndim)]

        std = [(disp[i] ** 2).cmean() ** 0.5 for i in range(delta.ndim)]

        if logger and pm.comm.rank == 0:
            logger.info("Displacement computed in configuration space: std = %s" % str(std))
    else:
        disp = None

    # return density and displacement (which could be None)
    return delta, disp


def lognormal_transform(density, bias=1.):
    r"""
    Apply a (biased) lognormal transformation of the density
    field by computing:

    .. math::

        F(\delta) = \frac{1}{N} e^{b*\delta}

    where :math:`\delta` is the initial overdensity field and the
    normalization :math:`N` is chosen such that
    :math:`\langle F(\delta) \rangle = 1`

    Parameters
    ----------
    density : array_like
        the input density field to apply the transformation to
    bias : float, optional
        optionally apply a linear bias to the density field;
        default is unbiased (1.0)

    Returns
    -------
    toret : RealField
        the real field holding the transformed density field
    """
    toret = density.copy()
    toret[:] = numpy.exp(bias * density.value)
    toret[:] /= toret.cmean(dtype='f8')

    return toret


def poisson_sample_to_points(delta, displacement, pm, nbar, bias=1., seed=None, logger=None):
    """
    Poisson sample the linear delta and displacement fields to points.

    The steps in this function:

    #.  Apply a biased, lognormal transformation to the input ``delta`` field
    #.  Poisson sample the overdensity field to discrete points
    #.  Disribute the positions of particles uniformly within the mesh cells,
        and assign the displacement field at each cell to the particles

    Parameters
    ----------
    delta : RealField
        the linear overdensity field to sample
    displacement : list of RealField (3,)
        the linear displacement fields which is used to move the particles
    nbar : float
        the desired number density of the output catalog of objects
    bias : float, optional
        apply a linear bias to the overdensity field (default is 1.)
    seed : int, optional
        the random seed used to Poisson sample the field to points

    Returns
    -------
    pos : array_like, (N, 3)
        the Cartesian positions of each of the generated particles
    displ : array_like, (N, 3)
        the displacement field sampled for each of the generated particles in the
        same units as the ``pos`` array
    """
    comm = delta.pm.comm

    # seed1 used for poisson sampling
    # seed2 used for uniform shift within a cell.
    seed1, seed2 = numpy.random.RandomState(seed).randint(0, 0xfffffff, size=2)

    # apply the lognormal transformation to the initial conditions density
    # this creates a positive-definite delta (necessary for Poisson sampling)
    lagrangian_bias = bias - 1.
    delta = lognormal_transform(delta, bias=lagrangian_bias)

    if logger and pm.comm.rank == 0:
        logger.info("Lognormal transformation done")

    # mean number of objects per cell
    H = delta.BoxSize / delta.Nmesh
    overallmean = H.prod() * nbar

    # number of objects in each cell (per rank, as a RealField)
    cellmean = delta * overallmean

    # create a random state with the input seed
    rng = MPIRandomState(seed=seed1, comm=comm, size=delta.size)

    # generate poissons. Note that we use ravel/unravel to
    # maintain MPI invariane.
    Nravel = rng.poisson(lam=cellmean.ravel())
    N = delta.pm.create(mode='real')
    N.unravel(Nravel)

    Ntot = N.csum()
    if logger and pm.comm.rank == 0:
        logger.info("Poisson sampling done, total number of objects is %d" % Ntot)

    pos_mesh = delta.pm.generate_uniform_particle_grid(shift=0.0)
    disp_mesh = numpy.empty_like(pos_mesh)

    # no need to do decompose because pos_mesh is strictly within the
    # local volume of the RealField.
    N_per_cell = N.readout(pos_mesh, resampler='nnb')
    for i in range(N.ndim):
        disp_mesh[:, i] = displacement[i].readout(pos_mesh, resampler='nnb')

    # fight round off errors, if any
    N_per_cell = numpy.int64(N_per_cell + 0.5)

    pos = pos_mesh.repeat(N_per_cell, axis=0)
    disp = disp_mesh.repeat(N_per_cell, axis=0)

    del pos_mesh
    del disp_mesh

    if logger and pm.comm.rank == 0:
        logger.info("catalog produced. Assigning in cell shift.")

    # generate linear ordering of the positions.
    # this should have been a method in pmesh, e.g. argument
    # to genereate_uniform_particle_grid(return_id=True);

    # FIXME: after pmesh update, remove this
    orderby = numpy.int64(pos[:, 0] / H[0] + 0.5)
    for i in range(1, delta.ndim):
        orderby[...] *= delta.Nmesh[i]
        orderby[...] += numpy.int64(pos[:, i] / H[i] + 0.5)

    # sort by ID to maintain MPI invariance.
    pos = mpsort.sort(pos, orderby=orderby, comm=comm)
    disp = mpsort.sort(disp, orderby=orderby, comm=comm)

    if logger and pm.comm.rank == 0:
        logger.info("sorting done")

    rng_shift = MPIRandomState(seed=seed2, comm=comm, size=len(pos))
    in_cell_shift = rng_shift.uniform(0, H[i], itemshape=(delta.ndim,))

    pos[...] += in_cell_shift
    pos[...] %= delta.BoxSize

    if logger and pm.comm.rank == 0:
        logger.info("catalog shifted.")

    return pos, disp
