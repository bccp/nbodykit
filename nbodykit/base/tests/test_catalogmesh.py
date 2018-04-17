from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import set_options
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import pytest

# debug logging
setup_logging("debug")


@MPITest([1, 4])
def test_gslice(comm):
    CurrentMPIComm.set(comm)

    # the catalog to slice
    d = UniformCatalog(1000, 1.0, seed=42)
    d = d.to_mesh(Nmesh=32)

    sels = [(0,10,1), (None,10,1), (0,50,4), (0,50,-1), (-10,-1,1), (-10,None,1)]
    for (start,stop,end) in sels:

        sliced = d.gslice(start,stop,end)

        for col in d:
            data1 = numpy.concatenate(comm.allgather(d[col]), axis=0)
            data2 = numpy.concatenate(comm.allgather(sliced[col]), axis=0)

            sl = slice(start,stop,end)
            assert_array_equal(data1[sl], data2, err_msg="using slice= "+str(sl))

    # empty slice
    sliced = d.gslice(0,0)
    assert len(sliced) == 0

@MPITest([1, 4])
def test_sort_ascending(comm):
    CurrentMPIComm.set(comm)

    # the mesh to sort
    d = UniformCatalog(100, 1.0)
    d['mass'] = 10**(d.rng.uniform(low=12, high=15, size=d.size))
    mesh = d.to_mesh(Nmesh=32)

    # invalid sort key
    with pytest.raises(ValueError):
        mesh2 = mesh.sort('BadColumn', reverse=False)

    # duplicate sort keys
    with pytest.raises(ValueError):
        mesh2 = mesh.sort(['mass', 'mass'])

    # sort in ascending order by mass
    mesh2 = mesh.sort('mass', reverse=False)

    # make sure we have all the columns
    assert all(col in mesh2 for col in mesh)

    mass = numpy.concatenate(comm.allgather(mesh['mass']))
    sorted_mass = numpy.concatenate(comm.allgather(mesh2['mass']))
    assert_array_equal(numpy.sort(mass), sorted_mass)


@MPITest([4])
def test_tsc_interlacing(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=3e-2, BoxSize=512., seed=42)

    # interlacing with TSC
    mesh = source.to_mesh(window='tsc', Nmesh=64, interlaced=True, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    r = FFTPower(mesh, mode='1d', kmin=0.02)

@MPITest([1])
def test_paint_chunksize(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=3e-2, BoxSize=512., seed=42)

    # interlacing with TSC
    mesh = source.to_mesh(window='tsc', Nmesh=64, interlaced=True, compensated=True)

    with set_options(paint_chunk_size=source.csize // 4):
        r1 = mesh.paint()

    with set_options(paint_chunk_size=source.csize):
        r2 = mesh.paint()

    assert_allclose(r1, r2)

@MPITest([4])
def test_cic_interlacing(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=3e-2, BoxSize=512., seed=42)

    # interlacing with TSC
    mesh = source.to_mesh(window='cic', Nmesh=64, interlaced=True, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    r = FFTPower(mesh, mode='1d', kmin=0.02)

@MPITest([4])
def test_setters(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=3e-2, BoxSize=512., seed=42)

    # make the mesh
    mesh = source.to_mesh(window='cic', Nmesh=64, interlaced=True, compensated=True)

    assert mesh.compensated == True
    mesh.compensated = False
    assert mesh.compensated == False

    assert mesh.interlaced == True
    mesh.interlaced = False
    assert mesh.interlaced == False

    assert mesh.window == 'cic'
    mesh.window = 'tsc'
    assert mesh.window == 'tsc'

@MPITest([4])
def test_bad_window(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=3e-2, BoxSize=512., seed=42)

    # make the mesh
    mesh = source.to_mesh(window='cic', Nmesh=64, interlaced=True, compensated=True)

    # no such window
    with pytest.raises(Exception):
        mesh.window = "BAD"

@MPITest([4])
def test_no_compensation(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=3e-2, BoxSize=512., seed=42)

    # make the mesh
    mesh = source.to_mesh(window='cic', Nmesh=64, interlaced=True, compensated=True)

    # no compensation for this window
    mesh.window = 'db6'

    # cannot compute compensation
    with pytest.raises(ValueError):
        actions = mesh.actions

@MPITest([1, 4])
def test_copy(comm):

    CurrentMPIComm.set(comm)
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)
    source['TEST'] = 10
    source.attrs['TEST'] = 'TEST'

    # make the mesh
    source = source.to_mesh(Nmesh=32)

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

@MPITest([4])
def test_slice(comm):
    CurrentMPIComm.set(comm)

    # the CatalogSource
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)
    source['TEST'] = 10.

    # the mesh
    source = source.to_mesh(Nmesh=32)

    # slice a subset
    subset = source[:10]
    assert all(col in subset for col in source.columns)
    assert isinstance(subset, source.__class__)
    assert len(subset) == 10
    assert_array_equal(subset['Position'], source['Position'].compute()[:10])

    subset = source[[0,1,2]]
    assert_array_equal(subset['Position'], source['Position'].compute()[[0,1,2]])

@MPITest([4])
def test_view(comm):
    CurrentMPIComm.set(comm)

    # the CatalogSource
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)
    source['TEST'] = 10.
    source.attrs['TEST'] = 10.0

    # the mesh
    mesh = source.to_mesh(Nmesh=32)

    # view
    view = mesh.view()
    assert view.base is mesh
    assert isinstance(view, mesh.__class__)

    # check meta-data
    for k in mesh.attrs:
        assert k in view.attrs

    # adding columns to the view changes original source
    view['TEST2'] = 5.0
    assert 'TEST2' in source
class CodeReached(BaseException): pass

@MPITest([1, 4])
def test_apply_nocompensation(comm):
    CurrentMPIComm.set(comm)

    # the CatalogSource
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)
    source['TEST'] = 10.
    source['Position2'] = source['Position']
    source.attrs['TEST'] = 10.0

    # the mesh
    mesh = source.to_mesh(position='Position2', Nmesh=32, compensated=False)

    def raisefunc(k, v):
        raise StopIteration

    mesh = mesh.apply(raisefunc)

    with pytest.raises(StopIteration):
        mesh.paint()

    # view
    view = mesh.view()
    assert view.base is mesh
    assert isinstance(view, mesh.__class__)

    # check meta-data
    for k in mesh.attrs:
        assert k in view.attrs

@MPITest([1])
def test_apply_compensated(comm):
    CurrentMPIComm.set(comm)

    # the CatalogSource
    source = UniformCatalog(nbar=0.2e-3, BoxSize=1024., seed=42)
    source['TEST'] = 10.
    source['Position2'] = source['Position']
    source.attrs['TEST'] = 10.0

    # the mesh
    mesh = source.to_mesh(position='Position2', Nmesh=32, compensated=True)

    def raisefunc(k, v):
        raise StopIteration

    mesh = mesh.apply(raisefunc)

    with pytest.raises(StopIteration):
        mesh.paint()

