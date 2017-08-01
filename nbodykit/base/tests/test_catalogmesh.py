from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import pytest

# debug logging
setup_logging("debug")

@MPITest([4])
def test_tsc_interlacing(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=3e-2, BoxSize=512., seed=42)

    # interlacing with TSC
    mesh = source.to_mesh(window='tsc', Nmesh=64, interlaced=True, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    r = FFTPower(mesh, mode='1d', kmin=0.02)

@MPITest([4])
def test_cic_interlacing(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=3e-2, BoxSize=512., seed=42)

    # interlacing with TSC
    mesh = source.to_mesh(window='cic', Nmesh=64, interlaced=True, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    r = FFTPower(mesh, mode='1d', kmin=0.02)
