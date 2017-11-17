from nbodykit import set_options, GlobalCache
import pytest

def test_bad_options():
    with pytest.raises(KeyError):
        set_options(no_this_option=3)


def test_cache_size():
    with set_options(global_cache_size=100):
        cache = GlobalCache.get()
        assert cache.cache.available_bytes == 100
