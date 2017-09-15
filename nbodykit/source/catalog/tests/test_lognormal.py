from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose

setup_logging("debug")

@MPITest([4])
def test_lognormal_sparse(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # this should generate 15 particles
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=1e-5, BoxSize=128., Nmesh=8, seed=42)

    mesh = source.to_mesh(compensated=False)

    real = mesh.paint(mode='real')
    assert_allclose(real.cmean(), 1.0)

@MPITest([1, 4])
def test_lognormal_dense(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=0.2e-2, BoxSize=128., Nmesh=8, seed=42)
    mesh = source.to_mesh(compensated=False)

    real = mesh.paint(mode='real')
    assert_allclose(real.cmean(), 1.0, rtol=1e-5)

@MPITest([1])
def test_lognormal_velocity(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=0.5e-2, BoxSize=1024., Nmesh=32, seed=42)

    source['Value'] = source['Velocity'][:, 0]**2
    mesh = source.to_mesh(compensated=False)

    real = mesh.paint(mode='real')
    velsum = comm.allreduce((source['Velocity'][:, 0]**2).sum().compute())
    velmean = velsum / source.csize

    assert_allclose(real.cmean(), velmean, rtol=1e-5)
