from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

try: import fastpm
except ImportError: fastpm = None

setup_logging()

@pytest.mark.skipif(fastpm is None, "fastpm is not installed")
def test_strong_scaling(benchmark):

    from fastpm.nbkit import FastPMCatalogSource

    # setup initial conditions
    cosmo = cosmology.Planck15
    power = cosmology.LinearPower(cosmo, 0)
    with benchmark("InitialConditions"):
        linear = LinearMesh(power, BoxSize=512, Nmesh=512)

    # run a computer simulation!
    with benchmark("Simulation"):
        sim = FastPMCatalogSource(linear, Nsteps=10)

    # run FOF to identify halo groups
    with benchmark("FOF"):
        fof = FOF(sim, 0.2, nmin=20)
        halos = fof.to_halos(1e12, cosmo, 0.)

    # populate halos with galaxies
    with benchmark("HOD"):
        hod = halos.populate(Zheng07Model)

    # compute and save galaxy power spectrum
    # result consistent on all ranks
    with benchmark("FFTPower")
        r = FFTPower(hod, mode='1d', Nmesh=128)
        r.save('galaxy-power.json')
