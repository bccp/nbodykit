from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import pytest

# debug logging
setup_logging("debug")

@MPITest([1])
def test_fftcorr_poles(comm):

    source = UniformCatalog(nbar=3e-3, BoxSize=512., seed=42, comm=comm)

    r = FFTCorr(source, mode='2d', BoxSize=1024, Nmesh=32, poles=[0,2,4])
    pkmu = r.corr['corr'].real
    mono = r.poles['corr_0'].real

    modes_1d = r.corr['modes'].sum(axis=-1)
    mono_from_pkmu = numpy.nansum(pkmu*r.corr['modes'], axis=-1) / modes_1d

    assert_array_equal(modes_1d, r.poles['modes'])
    assert_allclose(mono_from_pkmu, mono)

@MPITest([1])
def test_fftcorr_unique(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    r = FFTCorr(source, mode='1d', Nmesh=32, dr=0)
    p = r.corr
    assert_allclose(p.coords['r'], p['r'], rtol=1e-6)

@MPITest([1])
def test_fftcorr_padding(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    r = FFTCorr(source, mode='1d', BoxSize=1024, Nmesh=32)
    assert r.attrs['N1'] != 0
    assert r.attrs['N2'] != 0

@MPITest([1])
def test_fftcorr_save(comm):

    source = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)

    r = FFTCorr(source, mode='2d', Nmesh=32)
    r.save('fftcorr-test.json')

    r2 = FFTCorr.load('fftcorr-test.json', comm=comm)

    assert_array_equal(r.corr['r'], r2.corr['r'])
    assert_array_equal(r.corr['corr'], r2.corr['corr'])
    assert_array_equal(r.corr['mu'], r2.corr['mu'])
    assert_array_equal(r.corr['modes'], r2.corr['modes'])


@MPITest([1])
def test_fftcorr_mismatch_boxsize(comm):

    cosmo = cosmology.Planck15

    # input sources
    source1 = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-4, BoxSize=1024., seed=42, comm=comm)

    r = FFTCorr(source1, second=source2, mode='1d', BoxSize=1024, Nmesh=32)

@MPITest([1])
def test_fftcorr_mismatch_boxsize_fail(comm):

    cosmo = cosmology.Planck15

    # input sources
    mesh1 = UniformCatalog(nbar=3e-4, BoxSize=512., seed=42, comm=comm).to_mesh(Nmesh=32)
    mesh2 = UniformCatalog(nbar=3e-4, BoxSize=1024., seed=42, comm=comm).to_mesh(Nmesh=32)

    # raises an exception b/c meshes have different box sizes
    with pytest.raises(ValueError):
        r = FFTCorr(mesh1, second=mesh2, mode='1d', BoxSize=1024, Nmesh=32)

