from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose, assert_array_equal

setup_logging()

@MPITest([1,4])
def test_aamesh(comm):
    from nbodykit.source.mesh.aamesh import SSAAMesh

    cat = UniformCatalog(nbar=1e-2, BoxSize=(128, 128, 128), comm=comm)

    mesh = SSAAMesh(cat, Nmesh=(8, 8, 8))
    r = mesh.paint(mode='real')
    c = mesh.paint(mode='complex')

    assert_allclose(r.cmean(), 1.0)
