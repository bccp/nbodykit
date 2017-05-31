from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

import pytest

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_concatenate(comm):
    CurrentMPIComm.set(comm)

    # make two sources
    s1 = UniformCatalog(3e-6, 2600)
    s2 = UniformCatalog(3e-6, 2600)

    # concatenate all columns
    cat = transform.concatenate(s1, s2)

    # check the size and columns
    assert cat.size == s1.size + s2.size
    assert set(cat.columns) == set(s1.columns)

    # only one column
    cat = transform.concatenate(s1, s2, columns='Position')
    pos = numpy.concatenate([s1['Position'], s2['Position']], axis=0)
    numpy.testing.assert_array_equal(pos, cat['Position'])

    # fail on invalid column
    with pytest.raises(ValueError):
        cat = transform.concatenate(s1, s2, columns='InvalidColumn')
