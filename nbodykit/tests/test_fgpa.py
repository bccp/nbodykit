from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
from nbodykit.algorithms.fgpa import FGPA

# debug logging
setup_logging("debug")
    
@MPITest([1])
def test_fgpa(comm):
    CurrentMPIComm.set(comm)
    cosmo = cosmology.Planck15

    source2 = Source.LinearMesh(cosmology.NoWiggleEHPower(cosmo, 0.55), BoxSize=1.0, Nmesh=32, seed=33)

    fgpa = FGPA(source2, 1.0, 1.0, 100, 0)

