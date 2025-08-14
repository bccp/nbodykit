from nbodykit.lab import *
from nbodykit import set_options
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import pytest
from mpi4py import MPI
# debug logging
setup_logging("debug")

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_tsc_interlacing(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    # interlacing with TSC
    mesh = source.to_mesh(resampler='tsc', Nmesh=64, interlaced=True, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    r = FFTPower(mesh, mode='1d', kmin=0.02)
    # skip a few large scale modes that are noisier (fewer modes)
    assert_allclose(r.power['power'][5:], 1 / (3e-4), rtol=1e-1)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_paint_empty(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    source = source[:0]
    assert source.csize == 0

    # interlacing with TSC
    mesh = source.to_mesh(resampler='tsc', Nmesh=64, interlaced=True, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    real = mesh.to_real_field(normalize=True)
    assert_allclose(real, 1.0)

    real = mesh.to_real_field(normalize=False)
    assert_allclose(real, 0.0)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_paint_chunksize(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    # interlacing with TSC
    mesh = source.to_mesh(resampler='tsc', Nmesh=64, interlaced=True, compensated=True)

    with set_options(paint_chunk_size=source.csize // 4):
        r1 = mesh.compute()

    with set_options(paint_chunk_size=source.csize):
        r2 = mesh.compute()

    assert_allclose(r1, r2)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_shotnoise(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)
    source['Weight'] = source.rng.uniform()

    # interlacing with TSC
    mesh = source.to_mesh(resampler='tsc', Nmesh=64, interlaced=True, compensated=True, weight='Weight')

    with set_options(paint_chunk_size=source.csize // 4):
        r1 = mesh.compute()

    with set_options(paint_chunk_size=source.csize):
        r2 = mesh.compute()
    assert_allclose(r1, r2)

    # expected shotnoise for uniform weights between 0 and 1
    SN = 4 / 3.0 * 1 / (3e-4)
    assert_allclose(r1.attrs['shotnoise'], SN, rtol=1e-2)
    assert_allclose(r2.attrs['shotnoise'], SN, rtol=1e-2)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_cic_interlacing(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    # interlacing with TSC
    mesh = source.to_mesh(resampler='cic', Nmesh=64, interlaced=True, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    r = FFTPower(mesh, mode='1d', kmin=0.02)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_setters(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    # make the mesh
    mesh = source.to_mesh(resampler='cic', Nmesh=64, interlaced=True, compensated=True)

    assert mesh.compensated == True
    mesh.compensated = False
    assert mesh.compensated == False

    assert mesh.interlaced == True
    mesh.interlaced = False
    assert mesh.interlaced == False

    assert mesh.window == 'cic'
    mesh.window = 'tsc'
    assert mesh.window == 'tsc'

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_bad_window(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    # make the mesh
    mesh = source.to_mesh(resampler='cic', Nmesh=64, interlaced=True, compensated=True)

    # no such window
    with pytest.raises(Exception):
        mesh.window = "BAD"

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_no_compensation(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    # make the mesh
    mesh = source.to_mesh(resampler='cic', Nmesh=64, interlaced=True, compensated=True)

    # no compensation for this window
    mesh.window = 'db6'

    # cannot compute compensation
    with pytest.raises(ValueError):
        actions = mesh.actions

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_odd_chunksize(comm):
    # no errors shall occur. This is a regression test.

    source = ArrayCatalog({
        'Position': numpy.ones((2000, 3)),
    }, BoxSize=512., comm=comm)

    # make the mesh
    mesh = source.to_mesh(resampler='cic', Nmesh=64, interlaced=True, compensated=True)

    with set_options(paint_chunk_size=1111):
        mesh.compute()

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_view(comm):

    # the CatalogSource
    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)
    source['TEST'] = 10.
    source.attrs['TEST'] = 10.0

    # the mesh
    mesh = source.to_mesh(Nmesh=32)

    # view
    view = mesh.view()
    assert view.base is mesh
    from nbodykit.base.mesh import MeshSource
    assert isinstance(view, MeshSource)

    # check meta-data
    for k in mesh.attrs:
        assert k in view.attrs

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_apply_nocompensation(comm):

    # the CatalogSource
    source = UniformCatalog(nbar=2e-4, BoxSize=512, seed=42, comm=comm)
    source['TEST'] = 10.
    source['Position2'] = source['Position']
    source.attrs['TEST'] = 10.0

    # the mesh
    mesh = source.to_mesh(position='Position2', Nmesh=32, compensated=False)

    def raisefunc(k, v):
        raise StopIteration

    mesh = mesh.apply(raisefunc)

    with pytest.raises(StopIteration):
        mesh.compute()

    # view
    view = mesh.view()
    assert view.base is mesh
    assert isinstance(view, mesh.__class__)

    # check meta-data
    for k in mesh.attrs:
        assert k in view.attrs

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_apply_compensated(comm):

    # the CatalogSource
    source = UniformCatalog(nbar=2e-4, BoxSize=512., seed=42, comm=comm)
    source['TEST'] = 10.
    source['Position2'] = source['Position']
    source.attrs['TEST'] = 10.0

    # the mesh
    mesh = source.to_mesh(position='Position2', Nmesh=32, compensated=True)

    def raisefunc(k, v):
        raise StopIteration

    mesh = mesh.apply(raisefunc)

    with pytest.raises(StopIteration):
        mesh.compute()

