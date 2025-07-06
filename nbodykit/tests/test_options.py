from nbodykit import set_options, GlobalCache, use_distributed, use_mpi
import pytest
from mpi4py import MPI

def test_bad_options():
    with pytest.raises(KeyError):
        set_options(no_this_option=3)


def test_cache_size():
    with set_options(global_cache_size=100):
        cache = GlobalCache.get()
        assert cache.cache.available_bytes == 100

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_use_mpi(comm):
    use_mpi(comm)
