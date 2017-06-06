from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
import dask
dask.set_options(get=dask.get)
setup_logging("debug")

@MPITest([4])
def test_lognormal_sparse(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # this should generate 15 particles
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, 0.55),
                    nbar=1e-5, BoxSize=128., Nmesh=8, seed=42)

    mesh = source.to_mesh(compensated=False)

    real = mesh.paint(mode='real')
    assert_allclose(real.cmean(), 1.0)

@MPITest([1, 4])
def test_lognormal_dense(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, 0.55),
                    nbar=0.2e-2, BoxSize=128., Nmesh=8, seed=42)
    mesh = source.to_mesh(compensated=False)

    real = mesh.paint(mode='real')
    assert_allclose(real.cmean(), 1.0, rtol=1e-5)

@MPITest([1])
def test_lognormal_velocity(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, 0.55),
                nbar=0.5e-2, BoxSize=1024., Nmesh=32, seed=42)

    source['Weight'] = source['Velocity'][:, 0] ** 2

    mesh = source.to_mesh(compensated=False)

    real = mesh.paint(mode='real')
    velsum = comm.allreduce((source['Velocity'][:, 0]**2).sum().compute())
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

    source = ArrayCatalog(data, BoxSize=100, Nmesh=32)

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

    source = UniformCatalog(nbar=0.2e-2, BoxSize=1024., seed=42)
    source['Weight0'] = source['Velocity'][:, 0]
    source['Weight1'] = source['Velocity'][:, 1]
    source['Weight2'] = source['Velocity'][:, 2]

    mesh = source.to_mesh(Nmesh=128, compensated=True)

    assert_allclose(source['Position'], mesh['Position'])

    mesh = source.to_mesh(Nmesh=128, compensated=True, interlaced=True)

    assert_allclose(source['Position'], mesh['Position'])

    mesh = source.to_mesh(Nmesh=128, weight='Weight0')

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

    source = HDFCatalog(tmpfile, root='X', attrs={"Nmesh":32})

    assert_allclose(source['Position'], dset['Position'])

    os.unlink(tmpfile)

@MPITest([1, 4])
def test_dict(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)
    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )
    # use a dictionary
    data = dict(Position=data['Position'], Velocity=data['Velocity'])

    source = ArrayCatalog(data, BoxSize=100, Nmesh=32)

    assert source.csize == 100 * comm.size

    source['Velocity'] = source['Position'] + source['Velocity']

    source['Position'] = source['Position'] + source['Velocity']

    # Position triggers  Velocity which triggers Position and Velocity
    # which resolves to the true data.
    # so total is 3.
    assert_allclose(source['Position'], 3)

    mesh = source.to_mesh()
