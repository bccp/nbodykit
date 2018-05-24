from nbodykit.source.catalog.uniform import UniformCatalog
from runtests.mpi import MPITest
from nbodykit.lab import CurrentMPIComm
from nbodykit.utils import GatherArray
from numpy.testing import assert_array_equal
from mpi4py import MPI

@MPITest([4])
def test_uniform_invariant(comm):
    cat = UniformCatalog(nbar=2, BoxSize=100., seed=1234, dtype='f4', comm=comm)

    cat1 = UniformCatalog(nbar=2, BoxSize=100., seed=1234, dtype='f4', comm=MPI.COMM_SELF)

    allpos = GatherArray(cat['Position'].compute(), root=Ellipsis, comm=comm)

    assert_array_equal(allpos, cat1['Position'])

    allvel = GatherArray(cat['Velocity'].compute(), root=Ellipsis, comm=comm)

    assert_array_equal(allvel, cat1['Velocity'])
