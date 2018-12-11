from nbodykit import set_options, GlobalCache, use_distributed, use_mpi
import pytest
from runtests.mpi import MPITest

def test_bad_options():
    with pytest.raises(KeyError):
        set_options(no_this_option=3)


def test_cache_size():
    with set_options(global_cache_size=100):
        cache = GlobalCache.get()
        assert cache.cache.available_bytes == 100

@MPITest([1])
def test_use_distributed(comm):
    from distributed import LocalCluster, Client
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False)
    use_distributed(Client(cluster))

@MPITest([1, 4])
def test_use_mpi(comm):
    use_mpi(comm)
