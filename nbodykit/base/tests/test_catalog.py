from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging, set_options

import os
import pytest
from numpy.testing import assert_allclose, assert_array_equal

setup_logging()

@MPITest([1])
def test_default_columns(comm):
    CurrentMPIComm.set(comm)

    cat = UniformCatalog(nbar=100, BoxSize=1.0)

    # weight column is default
    assert cat['Weight'].is_default

    # override the default value --> no longer default
    cat['Weight'] = 10.
    assert not cat['Weight'].is_default


@MPITest([1, 4])
def test_save(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    import tempfile
    import shutil

    # initialize an output directory
    if comm.rank == 0:
        tmpfile = tempfile.mkdtemp()
    else:
        tmpfile = None
    tmpfile = comm.bcast(tmpfile)

    # initialize a uniform catalog
    source = UniformCatalog(nbar=0.2e-2, BoxSize=1024., seed=42)

    # add a non-array attrs (saved as JSON)
    source.attrs['empty'] = None

    # save to a BigFile
    source.save(tmpfile, source.columns)

    # assert that no default columns were saved
    datasets = os.listdir(tmpfile)
    assert not any(col in datasets for col in ['Value', 'Selection', 'Weight'])

    # load as a BigFileCatalog
    source2 = BigFileCatalog(tmpfile, attrs={"Nmesh":32})

    # check sources
    for k in source.attrs:
        assert_array_equal(source2.attrs[k], source.attrs[k])

    # check the data
    def allconcat(data):
        return numpy.concatenate(comm.allgather(data), axis=0)
    assert_allclose(allconcat(source['Position']), allconcat(source2['Position']))
    assert_allclose(allconcat(source['Velocity']), allconcat(source2['Velocity']))

    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(tmpfile)

@MPITest([1, 4])
def test_tomesh(comm):
    CurrentMPIComm.set(comm)

    source = UniformCatalog(nbar=0.2e-2, BoxSize=1024., seed=42)
    source['Weight0'] = source['Velocity'][:, 0]
    source['Weight1'] = source['Velocity'][:, 1]
    source['Weight2'] = source['Velocity'][:, 2]

    mesh = source.to_mesh(Nmesh=128, compensated=True)
    assert_allclose(source['Position'], mesh['Position'])

    mesh = source.to_mesh(Nmesh=128, compensated=True, interlaced=True)
    assert_allclose(source['Position'], mesh['Position'])

    mesh = source.to_mesh(Nmesh=128, weight='Weight0')

    # bad weight name
    with pytest.raises(ValueError):
        mesh = source.to_mesh(Nmesh=128, weight='Weight3')

    # bad window name
    with pytest.raises(ValueError):
        mesh = source.to_mesh(Nmesh=128, weight='Weight0', window='bad_window')

    # missing Nmesh
    with pytest.raises(ValueError):
        mesh = source.to_mesh()

    # missing BoxSize
    BoxSize = source.attrs.pop("BoxSize")
    with pytest.raises(ValueError):
        mesh = source.to_mesh(Nmesh=128)

@MPITest([4])
def test_bad_column(comm):
    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

    # read a missing column
    with pytest.raises(ValueError):
        data = source.read(['BAD_COLUMN'])

    # read a missing column
    with pytest.raises(ValueError):
        data = source.get_hardcolumn('BAD_COLUMN')

@MPITest([4])
def test_empty_slice(comm):
    CurrentMPIComm.set(comm)

    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

    # empty slice returns self
    source2 = source[source['Selection']]
    assert source is source2

    # non-empty selection on root only
    if comm.rank == 0:
        sel = source.rng.choice([True, False], size=source.size)
    else:
        sel = numpy.ones(source.size, dtype=bool)

    # this should trigger a full slice
    source2 = source[sel]
    assert source is not source2


@MPITest([4])
def test_slice(comm):
    CurrentMPIComm.set(comm)

    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

    # slice a subset
    subset = source[:10]
    assert all(col in subset for col in source.columns)
    assert isinstance(subset, source.__class__)
    assert len(subset) == 10
    assert_array_equal(subset['Position'], source['Position'].compute()[:10])

    subset = source[[0,1,2]]
    assert_array_equal(subset['Position'], source['Position'].compute()[[0,1,2]])

    # cannot slice with list of floats
    with pytest.raises(KeyError):
        subset = source[[0.0,1.0,2.0]]

    # missing column
    with pytest.raises(KeyError):
        col = source['BAD_COLUMN']

@MPITest([4])
def test_dask_slice(comm):
    CurrentMPIComm.set(comm)

    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

    # add a selection column
    index = numpy.random.choice([True, False], size=len(source))
    source['Selection'] = index

    # slice a column with a dask array
    pos = source['Position']
    pos2 = pos[source['Selection']]
    assert_array_equal(pos.compute()[index], pos2.compute())

@MPITest([1, 4])
def test_index(comm):
    CurrentMPIComm.set(comm)

    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)
    r = numpy.concatenate(comm.allgather(source.Index.compute()))
    assert_array_equal(r, range(source.csize))

    source = source.gslice(0, 1000)
    r = numpy.concatenate(comm.allgather(source.Index.compute()))
    assert_array_equal(r, range(source.csize))
    assert source.Index.dtype == numpy.dtype('i8')

@MPITest([1 ,4])
def test_transform(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)
    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )

    source = ArrayCatalog(data, BoxSize=100, Nmesh=32)

    source['Velocity'] = source['Position'] + source['Velocity']

    source['Position'] = source['Position'] + source['Velocity']

    # Position triggers  Velocity which triggers Position and Velocity
    # which resolves to the true data.
    # so total is 3.
    assert_allclose(source['Position'], 3)

    mesh = source.to_mesh()

@MPITest([1, 4])
def test_getitem_columns(comm):
    CurrentMPIComm.set(comm)

    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

    # bad column name
    with pytest.raises(KeyError):
        subset = source[['Position', 'BAD_COLUMN']]

    subset = source[['Position']]

    for col in subset:
        assert_array_equal(subset[col].compute(), source[col].compute())

@MPITest([1, 4])
def test_delitem(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

    # add a test column
    test = numpy.ones(source.size)
    source['test'] = test

    # cannot delete hard coded column
    with pytest.raises(ValueError):
        del source['Position']

    # cannot delete missing column
    with pytest.raises(ValueError):
        del source['BAD_COLUMN']

    assert 'test' in source
    del source['test']
    assert 'test' not in source

def test_columnaccessor():
    from nbodykit.base.catalog import ColumnAccessor
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

    c = source['Position']
    truth = c[0].compute()

    assert isinstance(c, ColumnAccessor)
    c *= 10.
    # c is no longer an accessor because it has transformed.
    assert not isinstance(c, ColumnAccessor)

    # thus it is not affecting original.
    assert_array_equal(source['Position'][0].compute(), truth)
    assert_array_equal(c[0].compute(), truth * 10.)

    # inplace modification is still OK
    source['Position'] *= 10
    assert_array_equal(source['Position'][0].compute(), truth * 10)

    # test pretty print.
    assert 'first' in str(source['Position'])
    assert 'last' in str(source['Position'])

    # test circular reference
    new_col = source['Selection']
    assert isinstance(new_col, ColumnAccessor)
    source['Selection2'] = new_col
    assert source['Selection'].catalog is source
    assert source['Selection2'].catalog is source

@MPITest([1, 4])
def test_copy(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)
    source['TEST'] = 10
    source.attrs['TEST'] = 'TEST'

    # store original data
    data = {}
    for col in source:
        data[col] = source[col].compute()

    # make copy
    copy = source.copy()

    # modify original
    source['Position'] += 100.
    source['Velocity'] *= 10.

    # check data is equal to original
    for col in copy:
        assert_array_equal(copy[col].compute(), data[col])

    # check meta-data
    for k in source.attrs:
        assert k in copy.attrs

    # adding columns to the copy doesn't add to original source
    copy['TEST2'] = 5.0
    assert 'TEST2' not in source

    # make sure attrs are independent.
    source.attrs['foo'] = 123
    assert 'foo' not in copy.attrs

@MPITest([4])
def test_view(comm):
    CurrentMPIComm.set(comm)

    # the CatalogSource
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)
    source['TEST'] = 10.
    source.attrs['TEST'] = 10.0

    # view
    view = source.view()
    assert view.base is source
    assert isinstance(view, source.__class__)

    # check meta-data
    for k in source.attrs:
        assert k in view.attrs

    # adding columns to the view changes original source
    view['TEST2'] = 5.0
    assert 'TEST2' in source

    # make sure attrs are dependent.
    source.attrs['foo'] = 123
    assert 'foo' in view.attrs
