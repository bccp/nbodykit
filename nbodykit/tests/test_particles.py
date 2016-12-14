from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose
import dask
dask.set_options(get=dask.get)
setup_logging("debug")

@MPITest([4])
def test_zeldovich_sparse(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = Source.ZeldovichParticles(cosmo, nbar=0.2e-6, redshift=0.55, BoxSize=128., Nmesh=8, rsd=[0, 0, 0], seed=42)

    source.compensated = False

    real = source.paint(mode='real')

    assert_allclose(real.cmean(), 1.0)

@MPITest([1, 4])
def test_zeldovich_dense(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = Source.ZeldovichParticles(cosmo, nbar=0.2e-2, redshift=0.55, BoxSize=128., Nmesh=8, rsd=[0, 0, 0], seed=42)

    source.compensated = False

    real = source.paint(mode='real')

    assert_allclose(real.cmean(), 1.0)

@MPITest([1])
def test_zeldovich_velocity(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = Source.ZeldovichParticles(cosmo, nbar=0.2e-2, redshift=0.55, BoxSize=1024., Nmesh=32, rsd=[0, 0, 0], seed=42)

    source.compensated = False

    source.set_transform({'Weight' : lambda x: x['Velocity'][:, 0]})

    real = source.paint(mode='real')
    velsum = comm.allreduce(source['Velocity'][:, 0].sum().compute())
    velmean = velsum / source.csize

    assert_allclose(real.cmean(), velmean, rtol=1e-5)

