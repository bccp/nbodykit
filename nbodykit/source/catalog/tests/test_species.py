from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_array_equal
import pytest

setup_logging()


@MPITest([1, 4])
def test_get_syntax(comm):

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)
    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)

    # test either get syntax
    test1 = numpy.random.random(size=len(source1))
    cat['data/test1'] = test1
    assert_array_equal(cat['data/test1'], cat['data']['test1'])

    test2 = numpy.random.random(size=len(source1))
    cat['data']['test2'] = test2
    assert_array_equal(cat['data/test2'], cat['data']['test2'])


@MPITest([1, 4])
def test_columns(comm):

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)

    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2, BoxSize=512., Nmesh=128)

    assert 'BoxSize' in cat.attrs
    assert 'Nmesh' in cat.attrs

    # check all columns are there
    for col in source1:
        assert 'data/' + col in cat
        assert 'randoms/' + col in cat

@MPITest([1, 4])
def test_bad_input(comm):

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)

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

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)

    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)

    for source, name in zip([source1, source2], ['data', 'randoms']):
        subcat = cat[name] # should be equal to source
        for col in source:
            assert col in subcat
            assert_array_equal(subcat[col].compute(), source[col].compute())
        for k in subcat.attrs:
            assert k in source.attrs


@MPITest([1, 4])
def test_setitem(comm):

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)

    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)

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
    with pytest.raises(ValueError):
        cat['data/test'] = test[:10]

@MPITest([1, 4])
def test_bad_slice(comm):

    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)

    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)

    # test with a column slice
    with pytest.raises(ValueError):
        subcat = cat[cat['data/Selection']]

    # test with a slice
    with pytest.raises(ValueError):
        subcat = cat[:10]
