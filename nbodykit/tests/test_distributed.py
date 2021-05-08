from runtests.mpi import MPITest
from nbodykit import set_options, GlobalCache, use_distributed, use_mpi
from numpy.testing import assert_array_equal
from nbodykit.lab import UniformCatalog
import pytest

def setup():
    from distributed import LocalCluster, Client
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False)
    use_distributed(Client(cluster))

def teardown():
    from mpi4py import MPI
    use_mpi(MPI.COMM_WORLD)

@MPITest([1])
def test_pickle(comm):
    import pickle
    cat = UniformCatalog(1e-3, 512, comm=comm)
    ss = pickle.dumps(cat)
    cat2 = pickle.loads(ss)
    
    assert_array_equal(cat['Position'], cat2['Position'])

import distributed

#@pytest.mark.xfail(distributed.__version__ in ['2021.03.0', '2021.04.0'],
#    reason="https://github.com/dask/distributed/issues/4565")
@MPITest([1])
def test_save(comm):
    cat = UniformCatalog(1e-3, 512, comm=comm)

    import tempfile
    import shutil

    tmpfile = tempfile.mkdtemp()
    cat.save(tmpfile, dataset='data')
    shutil.rmtree(tmpfile, ignore_errors=True)
