from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import pytest

# debug logging
setup_logging("debug")

@MPITest([1])
def test_fftrecon(comm):
    cosmo = cosmology.Planck15
    # this should generate 15 particles
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    CurrentMPIComm.set(comm)

    # should have used something better but this is fast enough
    data = LogNormalCatalog(Plin=Plin, bias=2, nbar=3e-3, BoxSize=512., Nmesh=64, seed=42)
    ran = UniformCatalog(nbar=3e-3, BoxSize=512., seed=42)

    mesh = FFTRecon(data=data, ran=ran, bias=2, Nmesh=64)

    r1 = FFTPower(mesh, mode='1d')
    r2 = FFTPower(data, mode='1d')

    # reconstruction shouldn't have matter much on large scale.
    assert_allclose(r1.power['power'][:5], r2.power['power'][:5], rtol=0.05)
