from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose
import dask
dask.set_options(get=dask.get)
setup_logging("debug")

@MPITest([1, 4])
def test_zeldovich(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = Source.ZeldovichParticles(cosmo, nbar=0.2e-6, redshift=0.55, BoxSize=128., Nmesh=8, rsd=[0, 0, 0], seed=42)

    source.compensated = False

    real = source.paint(mode='real')

    assert_allclose(real.cmean(), 1.0)

