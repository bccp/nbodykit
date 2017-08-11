from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_array_equal
import pytest

setup_logging()

@MPITest([1, 4])
def test_columns(comm):

    CurrentMPIComm.set(comm)

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84)

    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2, use_cache=True)

    # check all columns are there
    for col in source1:
        assert 'data/' + col in cat
        assert 'randoms/' + col in cat

@MPITest([1, 4])
def test_bad_input(comm):

    CurrentMPIComm.set(comm)

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84)

    # need 2 species
    with pytest.raises(ValueError):
        cat = MultipleSpeciesCatalog(['data'], source1)

    # non-unique names
    with pytest.raises(ValueError):
        cat = MultipleSpeciesCatalog(['data', 'data'], source1, source2)

    # mis-matched sizes
    with pytest.raises(ValueError):
        cat = MultipleSpeciesCatalog(['data', 'randoms', 'bad'], source1, source2)

    # bad comm
    with pytest.raises(ValueError):
        source1.comm = None
        cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)


@MPITest([1, 4])
def test_getitem(comm):

    CurrentMPIComm.set(comm)

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84)

    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2, use_cache=True)

    for source, name in zip([source1, source2], ['data', 'randoms']):
        subcat = cat[name] # should be equal to source
        for col in source:
            assert col in subcat
            assert_array_equal(subcat[col].compute(), source[col].compute())
        for k in subcat.attrs:
            assert k in source.attrs


@MPITest([1, 4])
def test_setitem(comm):

    CurrentMPIComm.set(comm)

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84)

    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2, use_cache=True)

    # bad name
    with pytest.raises(ValueError):
        cat['bad'] = numpy.ones(source1.size)

    # bad species name
    with pytest.raises(ValueError):
        cat['bad/test'] = numpy.ones(source1.size)

    test = numpy.ones(source1.size)*10
    cat['data/test'] = test
    assert_array_equal(cat['data/test'].compute(), test)

    # bad size
    with pytest.raises(AssertionError):
        cat['data/test'] = test[:10]
