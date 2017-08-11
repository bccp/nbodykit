from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

import pytest
from numpy.testing import assert_allclose, assert_array_equal

setup_logging()

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
    source.save(tmpfile, ['Position', 'Velocity'])

    # load as a BigFileCatalog
    source2 = BigFileCatalog(tmpfile, header='Header', attrs={"Nmesh":32})

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
    with pytest.raises(AttributeError):
        data = source.get_hardcolumn('BAD_COLUMN')

@MPITest([4])
def test_slice(comm):
    CurrentMPIComm.set(comm)

    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

    # slice a subset
    subset = source[:10]
    assert_array_equal(subset['Position'], source['Position'].compute()[:10])

    subset = source[[0,1,2]]
    assert_array_equal(subset['Position'], source['Position'].compute()[[0,1,2]])

    # cannot slice with list of floats
    with pytest.raises(KeyError):
        subset = source[[0.0,1.0,2.0]]

    # missing column
    with pytest.raises(KeyError):
        col = source['BAD_COLUMN']

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

@MPITest([1, 4])
def test_copy(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)

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
