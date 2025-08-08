from nbodykit.io.hdf import HDFFile
import os
import numpy
import tempfile
import pickle
import contextlib
import pytest
from mpi4py import MPI

# h5py is an optional dependency
try: import h5py
except ImportError: h5py = None

@contextlib.contextmanager
def temporary_data():
    """
    Write some temporary HDF5 data to disk
    """
    try:
        # generate data
        dset = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Mass', 'f8')])
        dset['Position'] = numpy.random.random(size=(1024, 3))
        dset['Mass'] = numpy.random.random(size=1024)

        # write to file
        tmpfile = tempfile.mkstemp()[1]
        with h5py.File(tmpfile , 'w') as ff:
            ff.create_dataset('X', data=dset) # store structured array as dataset
            grp = ff.create_group('Y')
            grp.create_dataset('Position', data=dset['Position']) # column as dataset
            grp.create_dataset('Mass', data=dset['Mass']) # column as dataset
            hdr = ff.create_dataset("Header", shape=(0,), dtype='f4')
            hdr.attrs['Value'] = 333
        yield (dset, tmpfile)
    except:
        raise
    finally:
        os.unlink(tmpfile)


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
@pytest.mark.skipif(h5py is None, reason="h5py is not installed")
def test_data(comm):

    with temporary_data() as (data, tmpfile):

        # read
        f = HDFFile(tmpfile, header='Header')

        # check columns
        cols = ['X/Mass', 'X/Position', 'Y/Mass', 'Y/Position']
        assert(all(col in cols for col in f.columns))

        assert f.attrs['Value'] == 333

        # make sure data is the same
        for col in cols:
            field = col.rsplit('/', 1)[-1]
            numpy.testing.assert_almost_equal(data[field], f[col][:])

        # try a slice
        data2 = f.read(cols, 0, 512, 1)
        for col in cols:
            field = col.rsplit('/', 1)[-1]
            numpy.testing.assert_almost_equal(data[field][:512], data2[col])

        # check size
        numpy.testing.assert_equal(f.size, 1024)


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
@pytest.mark.skipif(h5py is None, reason="h5py is not installed")
def test_nonzero_root(comm):

    with temporary_data() as (data, tmpfile):

        # read
        f = HDFFile(tmpfile, dataset='Y')

        # non-zero root
        f = HDFFile(tmpfile, dataset='Y')
        assert(all(col in ['Mass', 'Position'] for col in f.columns))

        # wrong root
        with pytest.raises(ValueError):
            f = HDFFile(tmpfile, dataset='Z')


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
@pytest.mark.skipif(h5py is None, reason="h5py is not installed")
def test_nonzero_exclude(comm):

    with temporary_data() as (data, tmpfile):

        # read
        f = HDFFile(tmpfile, exclude=['Y', 'Header'])

        # non-zero exclude
        f = HDFFile(tmpfile, exclude=['Y', 'Header'])
        assert(all(col in ['Mass', 'Position'] for col in f.columns))

        # non-zero exclude and root
        f = HDFFile(tmpfile, exclude=['Mass'], dataset='Y')
        assert all(col in ['Position'] for col in f.columns)

        # bad exclude
        with pytest.raises(ValueError):
            f = HDFFile(tmpfile, exclude=['Z'])


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
@pytest.mark.skipif(h5py is None, reason="h5py is not installed")
def test_data_mismatch(comm):

    # generate data
    pos = numpy.random.random(size=(1024, 3))
    mass = numpy.random.random(size=512)

    # write to file
    tmpfile = tempfile.mkstemp()[1]
    with h5py.File(tmpfile , 'w') as ff:
        ff.create_dataset('Mass', data=mass)
        ff.create_dataset('Position', data=pos)

    # fails due to mismatched sizes
    with pytest.raises(ValueError):
        f = HDFFile(tmpfile)

    # only one dataset now, so this works
    f = HDFFile(tmpfile, exclude=['Mass'])
    assert f.size == 1024

    os.unlink(tmpfile)


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
@pytest.mark.skipif(h5py is None, reason="h5py is not installed")
def test_empty(comm):

    # create empty file
    tmpfile = tempfile.mkstemp()[1]
    with h5py.File(tmpfile , 'w') as ff:
        ff.create_group('Y')

    # no datasets!
    with pytest.raises(ValueError):
        f = HDFFile(tmpfile)

    os.unlink(tmpfile)

