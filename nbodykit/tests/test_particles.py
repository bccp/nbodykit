from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
import dask
dask.set_options(get=dask.get)
setup_logging("debug")

@MPITest([4])
def test_zeldovich_sparse(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = Source.ZeldovichParticles(cosmo, nbar=0.2e-6, redshift=0.55, BoxSize=128., Nmesh=8, rsd=[0, 0, 0], seed=42)

    mesh = source.to_mesh()
    mesh.compensated = False

    real = mesh.paint(mode='real')

    assert_allclose(real.cmean(), 1.0)

@MPITest([1, 4])
def test_zeldovich_dense(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = Source.ZeldovichParticles(cosmo, nbar=0.2e-2, redshift=0.55, BoxSize=128., Nmesh=8, rsd=[0, 0, 0], seed=42)
    mesh = source.to_mesh()

    mesh.compensated = False

    real = mesh.paint(mode='real')

    assert_allclose(real.cmean(), 1.0)

@MPITest([1])
def test_zeldovich_velocity(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = Source.ZeldovichParticles(cosmo, nbar=0.2e-2, redshift=0.55, BoxSize=1024., Nmesh=32, rsd=[0, 0, 0], seed=42)

    source['Weight'] = source['Velocity'][:, 0]

    mesh = source.to_mesh()
    mesh.compensated = False

    real = mesh.paint(mode='real')
    velsum = comm.allreduce(source['Velocity'][:, 0].sum().compute())
    velmean = velsum / source.csize

    assert_allclose(real.cmean(), velmean, rtol=1e-5)

@MPITest([1])
def test_transform(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)
    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )

    source = Source.Array(data, BoxSize=100, Nmesh=32)

    source['Velocity'] = source['Position'] + source['Velocity']

    source['Position'] = source['Position'] + source['Velocity']

    # Position triggers  Velocity which triggers Position and Velocity
    # which resolves to the true data.
    # so total is 3.
    assert_allclose(source['Position'], 3)

    mesh = source.to_mesh()

@MPITest([1, 4])
def test_tomesh(comm):
    CurrentMPIComm.set(comm)

    source = Source.UniformParticles(nbar=0.2e-2, BoxSize=1024., seed=42)
    mesh = source.to_mesh(Nmesh=128)

    assert_allclose(source['Position'], mesh['Position'])

@MPITest([1, 4])
def test_save(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)
    import tempfile
    import shutil
    from nbodykit.io.bigfile import BigFile
    if comm.rank == 0:
        tmpfile = tempfile.mkdtemp()
    else:
        tmpfile = None

    tmpfile = comm.bcast(tmpfile)

    source = Source.UniformParticles(nbar=0.2e-2, BoxSize=1024., seed=42)

    source.save(tmpfile, ['Position', 'Velocity'])

    source2 = Source.File(BigFile, tmpfile, Nmesh=32, args=dict(header='Header'))

    for k in source.attrs:
        assert_array_equal(source2.attrs[k], source.attrs[k])

    def allconcat(data):
        return numpy.concatenate(comm.allgather(data), axis=0)

    assert_allclose(allconcat(source['Position']), allconcat(source2['Position']))
    assert_allclose(allconcat(source['Velocity']), allconcat(source2['Velocity']))

    comm.barrier()

    if comm.rank == 0:
        shutil.rmtree(tmpfile)

@MPITest([1])
def test_file(comm):
    import h5py
    import tempfile
    from nbodykit.io.hdf import HDFFile
    import os

    # fake structured array
    dset = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Mass', 'f8')])
    dset['Position'] = numpy.random.random(size=(1024, 3))
    dset['Mass'] = numpy.random.random(size=1024)

    tmpfile = tempfile.mkstemp()[1]
    
    with h5py.File(tmpfile , 'w') as ff:
        ds = ff.create_dataset('X', data=dset) # store structured array as dataset
        ds.attrs['BoxSize'] = 1.0
        grp = ff.create_group('Y')
        grp.create_dataset('Position', data=dset['Position']) # column as dataset
        grp.create_dataset('Mass', data=dset['Mass']) # column as dataset

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = Source.File(HDFFile, tmpfile, Nmesh=32, args={'root': 'X'})

    assert_allclose(source['Position'], dset['Position'])

    os.unlink(tmpfile)
