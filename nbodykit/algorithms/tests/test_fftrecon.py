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

    # very high bias to increase the accuracy of reconstruction;
    # since shotnoise is high.
    data = LogNormalCatalog(Plin=Plin, bias=4, nbar=1e-4, BoxSize=1024., Nmesh=64, seed=42, comm=comm)
    ran = UniformCatalog(nbar=1e-4, BoxSize=1024., seed=42, comm=comm)

    # lognormal mocks don't have the correct small scale power for reconstruction,
    # so we heavily smooth and assert the reconstruction
    # doesn't mess it all up.

    mesh = FFTRecon(data=data, ran=ran, bias=4, Nmesh=64, R=40)

    r1 = FFTPower(mesh, mode='1d')
    r2 = FFTPower(data, mode='1d')

    # reconstruction shouldn't have matter much on large scale.
    assert_allclose(r1.power['power'][:5], r2.power['power'][:5], rtol=0.05)

