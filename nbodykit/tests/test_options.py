from nbodykit import set_options
import pytest

def test_bad_options():
    with pytest.raises(KeyError):
        set_options(no_this_option=3)
