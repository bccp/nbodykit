from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose
import tempfile
import os

@MPITest([1])
def test_hdf(comm):

    import h5py

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

    source = HDFCatalog(tmpfile, root='X', attrs={"Nmesh":32}, comm=comm)
    assert_allclose(source['Position'], dset['Position'])

    os.unlink(tmpfile)

@MPITest([1])
def test_csv(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # read nrows
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVCatalog(ff.name, names, blocksize=100, comm=comm)

        # make sure data is the same
        for i, name in enumerate(names):
            numpy.testing.assert_almost_equal(data[:,i], f[name].compute(), decimal=7)

        # make sure all the columns are there
        assert all(col in f for col in names)

@MPITest([1])
def test_stack_glob(comm):

    tmpfile1 = 'test-glob-1.dat'
    tmpfile2 = 'test-glob-2.dat'

    # generate data
    data = numpy.random.random(size=(100,5))
    numpy.savetxt(tmpfile1, data, fmt='%.7e')
    numpy.savetxt(tmpfile2, data, fmt='%.7e')

    # read using a glob
    names =['a', 'b', 'c', 'd', 'e']
    f = CSVCatalog('test-glob-*', names, blocksize=100, comm=comm)

    # make sure print works
    print(f)

    # make sure data is the same
    fulldata = numpy.concatenate([data, data], axis=0)
    for i, name in enumerate(names):
        numpy.testing.assert_almost_equal(fulldata[:,i], f[name].compute(), decimal=7)

    # make sure all the columns are there
    assert all(col in f for col in names)

    os.unlink(tmpfile1)
    os.unlink(tmpfile2)

@MPITest([1])
def test_stack_list(comm):

    tmpfile1 = 'test-list-1.dat'
    tmpfile2 = 'test-list-2.dat'

    # generate data
    data = numpy.random.random(size=(100,5))
    numpy.savetxt(tmpfile1, data, fmt='%.7e')
    numpy.savetxt(tmpfile2, data, fmt='%.7e')

    # read using a glob
    names =['a', 'b', 'c', 'd', 'e']
    f = CSVCatalog(['test-list-1.dat', 'test-list-2.dat'], names, blocksize=100, comm=comm)

    # make sure print works
    print(f)

    # make sure data is the same
    fulldata = numpy.concatenate([data, data], axis=0)
    for i, name in enumerate(names):
        numpy.testing.assert_almost_equal(fulldata[:,i], f[name].compute(), decimal=7)

    # make sure all the columns are there
    assert all(col in f for col in names)

    os.unlink(tmpfile1)
    os.unlink(tmpfile2)
