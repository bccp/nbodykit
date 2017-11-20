from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

try: import fastpm
except ImportError: fastpm = None

setup_logging()

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
        halos = fof.to_halos(1e12, cosmo, 0.)

    with benchmark("FFTPower-Halo"):
        # compute and save halo P(k)
        r = FFTPower(halos, mode="1d", Nmesh=512)
        r.save("halos-power.json")

    # populate halos with galaxies
    with benchmark("HOD"):
        hod = halos.populate(Zheng07Model)

    # compute and save galaxy power spectrum
    # result consistent on all ranks
    with benchmark("FFTPower-Galaxy"):
        r = FFTPower(hod, mode='1d', Nmesh=512)
        r.save('galaxy-power.json')
