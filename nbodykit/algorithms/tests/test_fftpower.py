from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import pytest

# debug logging
setup_logging("debug")

@MPITest([4])
def test_tsc_aliasing(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)
    mesh = source.to_mesh(window='tsc', Nmesh=64, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    r = FFTPower(mesh, mode='1d', kmin=0.02)
    Pk = r.power['power'].real
    err = (2*Pk**2/r.power['modes'])**0.5

    # test chi2 < 1.0
    residual = (Pk-r.attrs['shotnoise'])/err
    red_chi2 = (residual**2).sum()/len(Pk) # should be about 0.5-0.6
    assert red_chi2 < 1.0

@MPITest([4])
def test_cic_aliasing(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)
    mesh = source.to_mesh(window='cic', Nmesh=64, compensated=True)

    # compute the power spectrum -- should be flat shot noise
    # if the compensation worked
    r = FFTPower(mesh, mode='1d', kmin=0.02)
    Pk = r.power['power'].real
    err = (2*Pk**2/r.power['modes'])**0.5

    # test chi2 < 1.0
    residual = (Pk-r.attrs['shotnoise'])/err
    red_chi2 = (residual**2).sum()/len(Pk) # should be about 0.5-0.6
    assert red_chi2 < 1.0


@MPITest([1])
def test_fftpower_poles(comm):

    source = UniformCatalog(nbar=3e-3, BoxSize=512., seed=42, comm=comm)

    r = FFTPower(source, mode='2d', BoxSize=1024, Nmesh=32, poles=[0,2,4])
    pkmu = r.power['power'].real
    mono = r.poles['power_0'].real

    modes_1d = r.power['modes'].sum(axis=-1)
    mono_from_pkmu = numpy.nansum(pkmu*r.power['modes'], axis=-1) / modes_1d

    assert_array_equal(modes_1d, r.poles['modes'])
    assert_allclose(mono_from_pkmu, mono)

@MPITest([1])
def test_fftpower_unique(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    r = FFTPower(source, mode='1d', Nmesh=32, dk=0)
    p = r.power
    assert_allclose(p.coords['k'], p['k'], rtol=1e-6)

@MPITest([1])
def test_fftpower_padding(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    r = FFTPower(source, mode='1d', BoxSize=1024, Nmesh=32)
    assert r.attrs['N1'] != 0
    assert r.attrs['N2'] != 0

@MPITest([1])
def test_fftpower_save(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    r = FFTPower(source, mode='2d', Nmesh=32)
    r.save('fftpower-test.json')

    r2 = FFTPower.load('fftpower-test.json', comm=comm)

    assert_array_equal(r.power['k'], r2.power['k'])
    assert_array_equal(r.power['power'], r2.power['power'])
    assert_array_equal(r.power['mu'], r2.power['mu'])
    assert_array_equal(r.power['modes'], r2.power['modes'])

@MPITest([1])
def test_fftpower(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    r = FFTPower(source, mode='1d', Nmesh=32)
    # the zero mode is cleared
    assert_array_equal(r.power['power'][0], 0)

@MPITest([1])
def test_fftpower_mismatch_boxsize(comm):

    cosmo = cosmology.Planck15

    # input sources
    source1 = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-4, BoxSize=1024., seed=42, comm=comm)

    r = FFTPower(source1, second=source2, mode='1d', BoxSize=1024, Nmesh=32)

@MPITest([1])
def test_fftpower_mismatch_boxsize_fail(comm):

    cosmo = cosmology.Planck15

    # input sources
    mesh1 = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm).to_mesh(Nmesh=32)
    mesh2 = UniformCatalog(nbar=3e-4, BoxSize=1024., seed=42, comm=comm).to_mesh(Nmesh=32)

    # raises an exception b/c meshes have different box sizes
    with pytest.raises(ValueError):
        r = FFTPower(mesh1, second=mesh2, mode='1d', BoxSize=1024, Nmesh=32)

@MPITest([1])
def test_projectedpower(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    Nmesh = 64
    rp1 = ProjectedFFTPower(source, Nmesh=Nmesh, axes=[1])

    # the zero mode is cleared
    assert_array_equal(rp1.power['power'][0], 0)

    rp2 = ProjectedFFTPower(source, Nmesh=Nmesh, axes=[0, 1])
    # the zero mode is cleared
    assert_array_equal(rp2.power['power'][0], 0)

    rf = FFTPower(source, Nmesh=Nmesh, mode='1d')

    # FIXME: why a factor of 2?
    assert_allclose(rp1.power['power'][1:].mean() * source.attrs['BoxSize'][0] ** 2, rf.power['power'][1:].mean(), rtol=2 * (Nmesh / 2)**-0.5)
    assert_allclose(rp2.power['power'][1:].mean() * source.attrs['BoxSize'][0], rf.power['power'][1:].mean(), rtol=2 * (Nmesh ** 2 / 2)**-0.5 * 10)
