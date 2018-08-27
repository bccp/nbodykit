from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from nbodykit.utils import GatherArray

from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from mpi4py import MPI

setup_logging("debug")

@MPITest([4])
def test_lognormal_sparse(comm):
    cosmo = cosmology.Planck15

    # this should generate 15 particles
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=1e-5, BoxSize=128., Nmesh=8, seed=42, comm=comm)

    mesh = source.to_mesh(compensated=False)

    real = mesh.compute(mode='real')
    assert_allclose(real.cmean(), 1.0, 1e-5)

@MPITest([1, 4])
def test_lognormal_dense(comm):
    cosmo = cosmology.Planck15

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=0.2e-2, BoxSize=128., Nmesh=8, seed=42, comm=comm)
    mesh = source.to_mesh(compensated=False)

    real = mesh.compute(mode='real')
    assert_allclose(real.cmean(), 1.0, rtol=1e-5)

@MPITest([4])
def test_lognormal_invariance(comm):
    cosmo = cosmology.Planck15

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=0.5e-2, BoxSize=128., Nmesh=32, seed=42, comm=comm)
    source1 = LogNormalCatalog(Plin=Plin, nbar=0.5e-2, BoxSize=128., Nmesh=32, seed=42, comm=MPI.COMM_SELF)

    assert source.csize == source1.size

    allpos = GatherArray(source['Position'].compute(), root=Ellipsis, comm=comm)
    assert_allclose(allpos, source1['Position'])
    alldis = GatherArray(source['Velocity'].compute(), root=Ellipsis, comm=comm)
    assert_allclose(alldis, source1['Velocity'])

@MPITest([1])
def test_lognormal_velocity(comm):
    cosmo = cosmology.Planck15

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=0.5e-2, BoxSize=128., Nmesh=32, seed=42, comm=comm)

    source['Value'] = source['Velocity'][:, 0]**2
    mesh = source.to_mesh(compensated=False)

    real = mesh.compute(mode='real')
    velsum = comm.allreduce((source['Velocity'][:, 0]**2).sum().compute())
    velmean = velsum / source.csize

    assert_allclose(real.cmean(), velmean, rtol=1e-5)
