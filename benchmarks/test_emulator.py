from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

try: import fastpm
except ImportError: fastpm = None

setup_logging()

def fof_catalog_with_benchmarks(benchmark, source, label, comm,
                                position='Position', velocity='Velocity',
                                initposition='InitialPosition',
                                peakcolumn=None, periodic=True):

    from nbodykit.utils import ScatterArray

    # make sure all of the columns are there
    for col in [position, velocity]:
        if col not in source:
            raise ValueError("the column '%s' is missing from parent source; cannot compute halos" %col)

    with benchmark("fof_catalog.count"):
        dtype=[('CMPosition', ('f4', 3)),('CMVelocity', ('f4', 3)),('Length', 'i4')]
        N = count(label, comm=comm)

    if periodic:
        # make sure BoxSize is there
        boxsize = source.attrs.get('BoxSize', None)
        if boxsize is None:
            raise ValueError("cannot compute halo catalog from source without 'BoxSize' in ``attrs`` dict")
    else:
        boxsize = None

    # center of mass position
    with benchmark("fof_catalog.centerofmass-pos"):
        hpos = centerofmass(label, source.compute(source[position]), boxsize=boxsize, comm=comm)

    # center of mass velocity
    with benchmark("fof_catalog.centerofmass-vel"):
        hvel = centerofmass(label, source.compute(source[velocity]), boxsize=None, comm=comm)

    # center of mass initial position
    if initposition in source:
        with benchmark("fof_catalog.centerofmass-initpos"):
            dtype.append(('InitialPosition', ('f4', 3)))
            hpos_init = centerofmass(label, source.compute(source[initposition]), boxsize=boxsize, comm=comm)
            hpos_init

    if peakcolumn is not None:
        assert peakcolumn in source

        dtype.append(('PeakPosition', ('f4', 3)))
        dtype.append(('PeakVelocity', ('f4', 3)))

        density = source[peakcolumn].compute()
        dmax = equiv_class(label, density, op=numpy.fmax, dense_labels=True, minlength=len(N), identity=-numpy.inf)
        comm.Allreduce(MPI.IN_PLACE, dmax, op=MPI.MAX)
        # remove any non-peak particle from the labels
        label1 = label * (density >= dmax[label])

        # compute the center of mass on the new labels
        ppos = centerofmass(label1, source.compute(source[position]), boxsize=boxsize, comm=comm)
        pvel = centerofmass(label1, source.compute(source[velocity]), boxsize=None, comm=comm)

    dtype = numpy.dtype(dtype)
    if comm.rank == 0:
        catalog = numpy.empty(shape=len(N), dtype=dtype)

        catalog['CMPosition'] = hpos
        catalog['CMVelocity'] = hvel
        catalog['Length'] = N
        catalog['Length'][0] = 0
        if 'InitialPosition' in dtype.names:
            catalog['InitialPosition'] = hpos_init

        if peakcolumn is not None:
            catalog['PeakPosition'] = ppos
            catalog['PeakVelocity'] = pvel
    else:
        catalog = None

    with benchmark("fof_catalog.ScatterArray"):
        toret = ScatterArray(catalog, comm, root=0)
    return toret

@pytest.mark.skipif(fastpm is None, reason="fastpm is not installed")
def test_strong_scaling(benchmark):

    from fastpm.nbkit import FastPMCatalogSource

    # setup initial conditions
    cosmo = cosmology.Planck15
    power = cosmology.LinearPower(cosmo, 0)
    linear = LinearMesh(power, BoxSize=512, Nmesh=512)

    with benchmark("FFTPower-Linear"):
        # compute and save linear P(k)
        r = FFTPower(linear, mode="1d")
        r.save("linear-power.json")

    # run a computer simulation!
    with benchmark("Simulation"):
        sim = FastPMCatalogSource(linear, Nsteps=10)

    with benchmark("FFTPower-Matter"):
        # compute and save matter P(k)
        r = FFTPower(sim, mode="1d", Nmesh=512)
        r.save("matter-power.json")

    # run FOF to identify halo groups
    with benchmark("FOF"):
        fof = FOF(sim, 0.2, nmin=20)

    #with benchmark("FOF-fof_catalog"):
    halos = fof_catalog_with_benchmarks(benchmark, sim, fof.labels, fof.comm)

    # with benchmark("FFTPower-Halo"):
    #     # compute and save halo P(k)
    #     r = FFTPower(halos, mode="1d", Nmesh=512)
    #     r.save("halos-power.json")

    # # populate halos with galaxies
    # with benchmark("HOD"):
    #     hod = halos.populate(Zheng07Model)
    #
    # # compute and save galaxy power spectrum
    # # result consistent on all ranks
    # with benchmark("FFTPower-Galaxy"):
    #     r = FFTPower(hod, mode='1d', Nmesh=512)
    #     r.save('galaxy-power.json')
