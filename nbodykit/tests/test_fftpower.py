from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit.algorithms.fof import FOF, HaloFinder
from nbodykit import setup_logging

from nbodykit.testing import TestingPowerSpectrum

# debug logging
setup_logging("debug")

@MPITest([1])
def test_fftpower_padding(comm):
    CurrentMPIComm.set(comm)
    # zeldovich particles
    source = Source.UniformParticles(nbar=3e-3, BoxSize=512., seed=42)

    r = FFTPower(source, mode='1d', BoxSize=1024, Nmesh=32)

@MPITest([1])
def test_fftpower_padding(comm):
    CurrentMPIComm.set(comm)
    # zeldovich particles
    source = Source.UniformParticles(nbar=3e-3, BoxSize=512., seed=42)

    r = FFTPower(source, mode='1d', Nmesh=32)

@MPITest([1])
def test_fftpower_mismatch_boxsize(comm):
    CurrentMPIComm.set(comm)
    # zeldovich particles
    source1 = Source.UniformParticles(nbar=3e-3, BoxSize=512., seed=42)

    source2 = Source.LinearMesh(TestingPowerSpectrum, BoxSize=1024, Nmesh=32, seed=33)

    r = FFTPower(source1, second=source2, mode='1d', BoxSize=1024, Nmesh=32)
