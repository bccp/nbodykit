from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose

from nbodykit.algorithms.kdtree import KDDensity

# debug logging
setup_logging("debug")
    
@MPITest([1, 4])
def test_kddensity(comm):
    cosmo = cosmology.Planck15

    CurrentMPIComm.set(comm)

    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, 0.55),
                nbar=3e-3, BoxSize=512., Nmesh=128, seed=42)

    kdden = KDDensity(source)
    assert(kdden.density.size , source.size)
    print(kdden.density.max())
