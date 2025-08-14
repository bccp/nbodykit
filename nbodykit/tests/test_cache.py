from nbodykit import GlobalCache
from nbodykit.lab import UniformCatalog
import pytest

from mpi4py import MPI

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_cache(comm):
    cat = UniformCatalog(nbar=10000, BoxSize=1.0, comm=comm)
    cat['test'] = cat['Position'] ** 5
    test = cat['test'].compute()

    # cache should no longer be empty
    cache = GlobalCache.get()

    assert cache.cache.total_bytes > 0

    # resize
    cache.cache.available_bytes = 100
    cache.cache.shrink()

    assert cache.cache.total_bytes < 100
