from nbodykit.lab import *
from nbodykit import setup_logging, set_options

import os
import pytest
import pytest_mpi
from mpi4py import MPI
from numpy.testing import assert_allclose, assert_array_equal

setup_logging()

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_default_columns(comm):
    cat = UniformCatalog(nbar=100, BoxSize=1.0, comm=comm)

    # weight column is default
    assert cat['Weight'].is_default

    # override the default value --> no longer default
    cat['Weight'] = 10.
    assert not cat['Weight'].is_default

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_save_future(comm, mpi_tmp_path):

    cosmo = cosmology.Planck15

    tmpfile = str(mpi_tmp_path)

    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3)),
            ('Mass', ('f4', 1))]
            )

    data['Mass'] = numpy.arange(len(data)).reshape(data['Mass'].shape)
    data['Position'] = numpy.arange(len(data) * 3).reshape(data['Position'].shape)
    data['Velocity'] = numpy.arange(len(data) * 3).reshape(data['Velocity'].shape)

    import dask.array as da
    source = ArrayCatalog(data, BoxSize=100, Nmesh=32, comm=comm)
    source['Rogue'] = da.ones((3, len(data)), chunks=(1, 1)).T

    # add a non-array attrs (saved as JSON)
    source.attrs['empty'] = None

    # save to a BigFile
    d = source.save(tmpfile, dataset='1', compute=False)

    # load as a BigFileCatalog; only attributes are saved
    source2 = BigFileCatalog(tmpfile, dataset='1', comm=comm)

    # check sources
    for k in source.attrs:
        assert_array_equal(source2.attrs[k], source.attrs[k])

    da.compute(d)

    # reload as a BigFileCatalog, data is saved
    source2 = BigFileCatalog(tmpfile, dataset='1', comm=comm)

    # check the data
    def allconcat(data):
        return numpy.concatenate(comm.allgather(data), axis=0)

    assert_allclose(allconcat(source['Position']), allconcat(source2['Position']))
    assert_allclose(allconcat(source['Velocity']), allconcat(source2['Velocity']))
    assert_allclose(allconcat(source['Mass']), allconcat(source2['Mass']))

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_save_dataset(comm, mpi_tmp_path):

    cosmo = cosmology.Planck15

    tmpfile = str(mpi_tmp_path)

    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3)),
            ('Mass', ('f4', 1))]
            )

    data['Mass'] = numpy.arange(len(data)).reshape(data['Mass'].shape)
    data['Position'] = numpy.arange(len(data) * 3).reshape(data['Position'].shape)
    data['Velocity'] = numpy.arange(len(data) * 3).reshape(data['Velocity'].shape)

    import dask.array as da
    source = ArrayCatalog(data, BoxSize=100, Nmesh=32, comm=comm)
    source['Rogue'] = da.ones((3, len(data)), chunks=(1, 1)).T

    subsample = source[::4]

    # add a non-array attrs (saved as JSON)
    source.attrs['empty'] = None

    # save to a BigFile
    source.save(tmpfile, dataset='1')

    # load as a BigFileCatalog
    source2 = BigFileCatalog(tmpfile, dataset='1', comm=comm)

    # check sources
    for k in source.attrs:
        assert_array_equal(source2.attrs[k], source.attrs[k])

    # check the data
    def allconcat(data):
        return numpy.concatenate(comm.allgather(data), axis=0)

    assert_allclose(allconcat(source['Position']), allconcat(source2['Position']))
    assert_allclose(allconcat(source['Velocity']), allconcat(source2['Velocity']))
    assert_allclose(allconcat(source['Mass']), allconcat(source2['Mass']))

    subsample.save(tmpfile, dataset='2')
    subsample2 = BigFileCatalog(tmpfile, dataset='2', comm=comm)

    assert_allclose(allconcat(subsample['Position']), allconcat(subsample2['Position']))
    assert_allclose(allconcat(subsample['Velocity']), allconcat(subsample2['Velocity']))
    assert_allclose(allconcat(subsample['Mass']), allconcat(subsample2['Mass']))

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_save(comm, mpi_tmp_path):

    cosmo = cosmology.Planck15

    tmpfile = str(mpi_tmp_path)

    # initialize a uniform catalog
    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

    # add a non-array attrs (saved as JSON)
    source.attrs['empty'] = None

    # save to a BigFile
    source.save(tmpfile, source.columns)

    # assert that no default columns were saved
    datasets = os.listdir(tmpfile)
    assert not any(col in datasets for col in ['Value', 'Selection', 'Weight'])

    # load as a BigFileCatalog
    source2 = BigFileCatalog(tmpfile, attrs={"Nmesh":32}, comm=comm)

    # check sources
    for k in source.attrs:
        assert_array_equal(source2.attrs[k], source.attrs[k])

    # check the data
    def allconcat(data):
        return numpy.concatenate(comm.allgather(data), axis=0)
    assert_allclose(allconcat(source['Position']), allconcat(source2['Position']))
    assert_allclose(allconcat(source['Velocity']), allconcat(source2['Velocity']))

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_tomesh(comm):

    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)
    source['Weight0'] = source['Velocity'][:, 0]
    source['Weight1'] = source['Velocity'][:, 1]
    source['Weight2'] = source['Velocity'][:, 2]

    mesh = source.to_mesh(Nmesh=128, compensated=True)

    mesh = source.to_mesh(Nmesh=128, compensated=True, interlaced=True)

    mesh = source.to_mesh(Nmesh=128, weight='Weight0')

    # bad weight name
    with pytest.raises(ValueError):
        mesh = source.to_mesh(Nmesh=128, weight='Weight3')

    # bad resampler name
    with pytest.raises(ValueError):
        mesh = source.to_mesh(Nmesh=128, weight='Weight0', resampler='bad_window')

    # missing Nmesh
    with pytest.raises(ValueError):
        mesh = source.to_mesh()

    # missing BoxSize
    BoxSize = source.attrs.pop("BoxSize")
    with pytest.raises(ValueError):
        mesh = source.to_mesh(Nmesh=128)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_bad_column(comm):
    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

    # read a missing column
    with pytest.raises(ValueError):
        data = source.read(['BAD_COLUMN'])

    # read a missing column
    with pytest.raises(ValueError):
        data = source.get_hardcolumn('BAD_COLUMN')

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_empty_slice(comm):

    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

    # Ellipsis slice returns self
    source2 = source[...]
    assert source is source2

    # Empty slice dos not crash
    subset = source[[]]
    assert all(col in subset for col in source.columns)
    assert isinstance(subset, source.__class__)

    # any selection on root only
    sel = source.rng.choice([True, False])
    if comm.rank != 0:
        sel[...] = True

    # this should trigger a full slice
    source2 = source[sel]
    assert source is not source2


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_slice(comm):

    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

    source['NZ'] = 1
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_dask_slice(comm):

    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

    # add a selection column
    index = numpy.random.choice([True, False], size=len(source))
    source['Selection'] = index

    # slice a column with a dask array
    pos = source['Position']
    pos2 = pos[source['Selection']]
    assert_array_equal(pos.compute()[index], pos2.compute())

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_index(comm):

    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)
    r = numpy.concatenate(comm.allgather(source.Index.compute()))
    assert_array_equal(r, range(source.csize))

    source = source.gslice(0, 1000)
    assert source.comm.size == comm.size

    r = numpy.concatenate(comm.allgather(source.Index.compute()))
    assert_array_equal(r, range(source.csize))
    assert source.Index.dtype == numpy.dtype('i8')

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_transform(comm):
    cosmo = cosmology.Planck15

    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )

    source = ArrayCatalog(data, BoxSize=100, Nmesh=32, comm=comm)

    source['Velocity'] = source['Position'] + source['Velocity']

    source['Position'] = source['Position'] + source['Velocity']

    # Position triggers  Velocity which triggers Position and Velocity
    # which resolves to the true data.
    # so total is 3.
    assert_allclose(source['Position'], 3)

    mesh = source.to_mesh()

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_getitem_columns(comm):

    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

    # bad column name
    with pytest.raises(KeyError):
        subset = source[['Position', 'BAD_COLUMN']]

    subset = source[['Position']]

    for col in subset:
        assert_array_equal(subset[col].compute(), source[col].compute())

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_delitem(comm):

    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_columnaccessor(comm):
    from nbodykit.base.catalog import ColumnAccessor
    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_copy(comm):

    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)
    source['TEST'] = 10
    source.attrs['TEST'] = 'TEST'

    # store original data
    data = {}
    for col in source:
        data[col] = source[col].compute()

    # make copy
    copy = source.copy()

    assert copy.comm.size == comm.size

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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_view(comm):
    # the CatalogSource
    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)
    source['TEST'] = 10.
    source.attrs['TEST'] = 10.0

    # view
    view = source.view()
    assert view.comm.size == comm.size

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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_persist(comm):
    # the CatalogSource
    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)
    source1 = source.persist(columns=['Position'])

    for key in source1.columns:
        assert_allclose(source[key], source1[key])

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sort(comm):
    # the CatalogSource
    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)

    source['ranks'] = numpy.float32(source.csize - source.Index)
    s = source.sort('ranks')

    arr = numpy.concatenate(comm.allgather(s['ranks'].compute()))
    assert (numpy.diff(arr) > 0).all()
