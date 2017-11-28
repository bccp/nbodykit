from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

import pytest
import tempfile
import shutil
from numpy.testing import assert_allclose, assert_array_equal

setup_logging()

@MPITest([1,4])
def test_lost_attrs(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # initialize an output directory
    if comm.rank == 0:
        tmpfile = tempfile.mkdtemp()
    else:
        tmpfile = None
    tmpfile = comm.bcast(tmpfile)

    # linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42)

    # a hard to save attribute
    source.attrs['bad'] = cosmo.to_astropy()

    # generate a warning due to lost attr
    with pytest.warns(UserWarning):
        source.save(tmpfile, mode='real')

    # cleanup
    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(tmpfile)

@MPITest([1,4])
def test_real_save(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # initialize an output directory
    if comm.rank == 0:
        tmpfile = tempfile.mkdtemp()
    else:
        tmpfile = None
    tmpfile = comm.bcast(tmpfile)

    # linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42)

    # a hard to save attribute
    source.attrs['empty'] = None

    # save to bigfile
    source.save(tmpfile, mode='complex')

    # load as a BigFileMesh
    source2 = BigFileMesh(tmpfile, dataset='Field')

    # check sources
    for k in source.attrs:
        assert_array_equal(source2.attrs[k], source.attrs[k])

    # check data
    assert_array_equal(source2.paint(mode='complex'), source.paint(mode='complex'))

    # cleanup
    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(tmpfile)

@MPITest([1,4])
def test_real_save(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # initialize an output directory
    if comm.rank == 0:
        tmpfile = tempfile.mkdtemp()
    else:
        tmpfile = None
    tmpfile = comm.bcast(tmpfile)

    # linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42)

    # a hard to save attribute
    source.attrs['empty'] = None

    # save to bigfile
    source.save(tmpfile, mode='real')

    # load as a BigFileMesh
    source2 = BigFileMesh(tmpfile, dataset='Field')

    # check sources
    for k in source.attrs:
        assert_array_equal(source2.attrs[k], source.attrs[k])

    # check data
    assert_array_equal(source2.paint(mode='real'), source.paint(mode='real'))

    # cleanup
    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(tmpfile)

@MPITest([1,4])
def test_preview(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42)

    # the painted RealField
    real = source.paint(mode='real')

    preview = source.preview()
    assert_allclose(preview.sum(), real.csum(), rtol=1e-5)

    real[...] **= 2
    preview[...] **= 2
    assert_allclose(preview.sum(), real.csum(), rtol=1e-5)

@MPITest([1,4])
def test_resample(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42)

    # re-sample to Nmesh=32
    real = source.paint(mode='real', Nmesh=32)

    # and preview at same resolution
    preview = source.preview(Nmesh=32)
    assert_allclose(preview.sum(), real.csum(), rtol=1e-5)

    # XXX: disabled because currently paint uses fourier space resample
    # and preview uses configuration resample.

    #real[...] **= 2
    #preview[...] **= 2
    #assert_allclose(preview.sum(), real.csum(), rtol=1e-5)

@MPITest([1])
def test_bad_mode(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42)

    with pytest.raises(ValueError):
        field = source.to_field(mode='BAD')

    with pytest.raises(ValueError):
        field = source.paint(mode='BAD')

@MPITest([4])
def test_view(comm):

    from nbodykit.base.mesh import MeshSource
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42)
    source.attrs['TEST'] = 10.0

    # view
    view = source.view()
    assert view.base is source
    assert isinstance(view, MeshSource)

    # check meta-data
    for k in source.attrs:
        assert k in view.attrs
