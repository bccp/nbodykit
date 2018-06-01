from runtests.mpi import MPITest
from nbodykit import setup_logging
from nbodykit.binned_statistic import BinnedStatistic

import pytest
import tempfile
import numpy.testing as testing
import numpy
import os

data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')
setup_logging("debug")


@MPITest([1])
def test_to_json(comm):

    # load from JSON
    ds1 = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_1d.json'))

    # to JSON
    with tempfile.NamedTemporaryFile(delete=False) as ff:
        ds1.to_json(ff.name)
    ds2 = BinnedStatistic.from_json(ff.name)

    # same data?
    for name in ds1:
        testing.assert_almost_equal(ds1[name], ds2[name])

    # cleanup
    os.remove(ff.name)

@MPITest([1])
def test_1d_load(comm):

    # load plaintext format
    with pytest.warns(FutureWarning):
        ds1 = BinnedStatistic.from_plaintext(['k'], os.path.join(data_dir, 'dataset_1d_deprecated.dat'))

        # wrong dimensions
        with pytest.raises(ValueError):
            ds1 = BinnedStatistic.from_plaintext(['k', 'mu'], os.path.join(data_dir, 'dataset_1d_deprecated.dat'))

    # load from JSON
    ds2 = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_1d.json'))

    # same data?
    for name in ds1:
        testing.assert_almost_equal(ds1[name], ds2[name])

@MPITest([1])
def test_2d_load(comm):

    # load plaintext format
    with pytest.warns(FutureWarning):
        ds1 = BinnedStatistic.from_plaintext(['k', 'mu'], os.path.join(data_dir, 'dataset_2d_deprecated.dat'))

    # load from JSON
    ds2 = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    # same data?
    for name in ds1:
        testing.assert_almost_equal(ds1[name], ds2[name])

@MPITest([1])
def test_str(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    # list all variable names
    s = str(dataset)

    # now just list total number of variables
    dataset['test1'] = numpy.ones(dataset.shape)
    dataset['test2'] = numpy.ones(dataset.shape)
    s = str(dataset)

    # this is the same as str
    r = repr(dataset)

@MPITest([1])
def test_getitem(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    # invalid key
    with pytest.raises(KeyError):
        bad = dataset['error']

    # slice columns
    sliced = dataset[['k', 'mu', 'power']]
    sliced = dataset[('k', 'mu', 'power')]

    # invalid slice
    with pytest.raises(KeyError):
        bad =dataset[['k', 'mu', 'error']]

    # too many dims in slice
    with pytest.raises(IndexError):
        bad = dataset[0,0,0]

    # cannot access single element of 2D power
    with pytest.raises(IndexError):
        bad = dataset[0,0]

@MPITest([1])
def test_array_slice(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    # get the first mu column
    sliced = dataset[:,0]
    assert sliced.shape[0] == dataset.shape[0]
    assert len(sliced.shape) == 1
    assert sliced.dims == ['k']

    # get the first mu column but keep dimension
    sliced = dataset[:,[0]]
    assert sliced.shape[0] == dataset.shape[0]
    assert sliced.shape[1] == 1
    assert sliced.dims == ['k', 'mu']


@MPITest([1])
def test_list_array_slice(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    # get the first and last mu column
    sliced = dataset[:,[0, -1]]
    assert len(sliced.shape) == 2
    assert sliced.dims == ['k', 'mu']

    # make sure we grabbed the right data
    for var in dataset:
        testing.assert_array_equal(dataset[var][:,[0,-1]], sliced[var])


@MPITest([1])
def test_variable_set(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))
    modes = numpy.ones(dataset.shape)

    # add new variable
    dataset['TEST'] = modes
    assert 'TEST' in dataset

    # override existing variable
    dataset['modes'] = modes
    assert numpy.all(dataset['modes'] == 1.0)

    # needs right shape
    with pytest.raises(ValueError):
        dataset['TEST'] = 10.


@MPITest([1])
def test_copy(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))
    copy = dataset.copy()
    for var in dataset:
        testing.assert_array_equal(dataset[var], copy[var])

@MPITest([1])
def test_rename_variable(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))
    test = numpy.zeros(dataset.shape)
    dataset['test'] = test

    dataset.rename_variable('test', 'renamed_test')
    assert 'renamed_test' in dataset
    assert 'test' not in dataset

@MPITest([1])
def test_sel(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    # no exact match fails
    with pytest.raises(IndexError):
        sliced = dataset.sel(k=0.1)

    # this should be squeezed
    sliced = dataset.sel(k=0.1, method='nearest')
    assert len(sliced.dims) == 1

    # this is not squeezed
    sliced = dataset.sel(k=[0.1], method='nearest')
    assert sliced.shape[0] == 1

    # this return empty k with arbitary edges.
    sliced = dataset.sel(k=[], method='nearest')
    assert sliced.shape[0] == 0

    # slice in a specific k-range
    sliced = dataset.sel(k=slice(0.02, 0.15), mu=[0.5], method='nearest')
    assert sliced.shape[1] == 1
    assert numpy.alltrue((sliced['k'] >= 0.02)&(sliced['k'] <= 0.15))

@MPITest([1])
def test_take(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    sliced = dataset.take(k=[8])
    assert sliced.shape[0] == 1
    assert len(sliced.dims) == 2

    sliced = dataset.take(k=[])
    assert sliced.shape[0] == 0
    assert len(sliced.dims) == 2

    dataset.take(k=dataset.coords['k'] < 0.3)
    assert len(sliced.dims) == 2

    dataset.take(dataset['modes'] > 0)
    assert len(sliced.dims) == 2

    dataset.take(dataset['k'] < 0.3)
    assert len(sliced.dims) == 2

@MPITest([1])
def test_squeeze(comm):

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    # need to specify which dimension to squeeze
    with pytest.raises(ValueError):
        squeezed = dataset.squeeze()
    with pytest.raises(ValueError):
        squeezed = dataset[[0],[0]].squeeze()

    sliced = dataset[:,[2]]
    with pytest.raises(ValueError):
        squeezed = sliced.squeeze('k')
    squeezed = sliced.squeeze('mu')

    assert len(squeezed.dims) == 1
    assert squeezed.shape[0] == sliced.shape[0]


@MPITest([1])
def test_average(comm):
    import warnings

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    # unweighted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        avg = dataset.average('mu')
        for var in dataset.variables:
            if var in dataset._fields_to_sum:
                x = numpy.nansum(dataset[var], axis=-1)
            else:
                x = numpy.nanmean(dataset[var], axis=-1)
            testing.assert_allclose(x, avg[var])

        # weighted
        weights = numpy.random.random(dataset.shape)
        dataset['weights'] = weights
        avg = dataset.average('mu', weights='weights')

        for var in dataset:
            if var in dataset._fields_to_sum:
                x = numpy.nansum(dataset[var], axis=-1)
            else:
                x = numpy.nansum(dataset[var]*dataset['weights'], axis=-1)
                x /= dataset['weights'].sum(axis=-1)
            testing.assert_allclose(x, avg[var])


@MPITest([1])
def test_reindex(comm):
    import warnings

    dataset = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    with pytest.raises(ValueError):
        new, spacing = dataset.reindex('k', 0.005, force=True, return_spacing=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        weights = numpy.random.random(dataset.shape)
        dataset['weights'] = weights
        new, spacing = dataset.reindex('k', 0.02, weights='weights', force=True, return_spacing=True)

        diff = numpy.diff(new.coords['k'])
        assert numpy.alltrue(diff > numpy.diff(dataset.coords['k'])[0])

        with pytest.raises(ValueError):
            new = dataset.reindex('mu', 0.4, force=False)
        new = dataset.reindex('mu', 0.4, force=True)

@MPITest([1])
def test_subclass_copy_sel(comm):
    # this test asserts the sel returns instance of subclass.
    # and the copy method can change the class.

    class A(BinnedStatistic):
        def mymethod(self):
            return self.copy(cls=BinnedStatistic)

    # load from JSON
    dataset = A.from_json(os.path.join(data_dir, 'dataset_2d.json'))

    dataset.mymethod()

    # no exact match fails
    with pytest.raises(IndexError):
        sliced = dataset.sel(k=0.1)

    # this should be squeezed
    sliced = dataset.sel(k=0.1, method='nearest')
    assert len(sliced.dims) == 1

    assert isinstance(sliced, A)
    assert isinstance(sliced.mymethod(), BinnedStatistic)

