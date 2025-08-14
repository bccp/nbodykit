from mpi4py import MPI
from nbodykit import use_distributed, use_mpi
from numpy.testing import assert_array_equal
from nbodykit.lab import UniformCatalog
import pytest
import pytest_mpi

def setup():
    # only initializes the local cluster this on the root rank.
    # all unit tests in this file must be protected by MPITest([1]).
    if MPI.COMM_WORLD.rank == 0:
        from distributed import LocalCluster, Client
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False)
        use_distributed(Client(cluster))

def teardown():
    use_mpi(MPI.COMM_WORLD)

#This fails with some but not all MPIs because we cannot always pickle the communicator
@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi_xfail
def test_pickle(comm):
    import pickle
    cat = UniformCatalog(1e-3, 512, comm=comm)
    ss = pickle.dumps(cat)
    cat2 = pickle.loads(ss)

    assert_array_equal(cat['Position'], cat2['Position'])

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_save(comm, mpi_tmp_path):
    cat = UniformCatalog(1e-3, 512, comm=comm)

    tmpfile = str(mpi_tmp_path)
    cat.save(tmpfile, dataset='data')
