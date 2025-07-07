from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
from mpi4py import MPI
import pytest
from nbodykit.algorithms.kdtree import KDDensity

# debug logging
setup_logging("debug")

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_kddensity(comm):
    cosmo = cosmology.Planck15

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=3e-4, BoxSize=64., Nmesh=16, seed=42, comm=comm)

    kdden = KDDensity(source)
    assert kdden.density.size == source.size
    print(kdden.density.max())
