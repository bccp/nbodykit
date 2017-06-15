from runtests.mpi import MPITest
from nbodykit.io.fits import FITSFile

import os
import numpy
import tempfile
import pickle
import contextlib
import pytest

try: import fitsio
except ImportError: fitsio is None

@contextlib.contextmanager
def temporary_data(data='table'):
    """
    Write some temporary FITS data to disk
    """
    try:
        # generate data
        if data == 'table':
            dset = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Mass', 'f8')])
            dset['Position'] = numpy.random.random(size=(1024, 3))
            dset['Mass'] = numpy.random.random(size=1024)
        else:
            dset = numpy.random.random(size=(1024, 3))

        # write to file
        tmpdir = tempfile.gettempdir()
        tmpfile = os.path.join(tmpdir, 'nbkit_tmp_data.fits')
        fitsio.write(tmpfile, dset, extname='Catalog')

        yield (dset, tmpfile)
    except:
        raise
    finally:
        os.unlink(tmpfile)


@MPITest([1])
@pytest.mark.skipif(fitsio is None, "fitsio is not installed")
def test_data(comm):

    with temporary_data() as (data, tmpfile):

        # read
        f = FITSFile(tmpfile)

        # check columns
        cols = ['Mass', 'Position']
        assert(all(col in cols for col in f.columns))

        # make sure data is the same
        for col in cols:
            numpy.testing.assert_almost_equal(data[col], f[col][:])

        # try a slice
        data2 = f.read(cols, 0, 512, 1)
        for col in cols:
            numpy.testing.assert_almost_equal(data[col][:512], data2[col])

        # check size
        numpy.testing.assert_equal(f.size, 1024)


@MPITest([1])
@pytest.mark.skipif(fitsio is None, "fitsio is not installed")
def test_string_ext(comm):

    with temporary_data() as (data, tmpfile):

        # read
        f = FITSFile(tmpfile, ext='Catalog')
        assert(all(col in ['Mass', 'Position'] for col in f.columns))
        assert(f.size == 1024)

        # read
        f = FITSFile(tmpfile, ext=1)
        assert(all(col in ['Mass', 'Position'] for col in f.columns))
        assert(f.size == 1024)

        # wrong ext
        with pytest.raises(ValueError):
            f = FITSFile(tmpfile, ext='WrongName')

@MPITest([1])
@pytest.mark.skipif(fitsio is None, "fitsio is not installed")
def test_wrong_ext(comm):

    with temporary_data() as (data, tmpfile):

        # no binary table data at 0
        with pytest.raises(ValueError):
            f = FITSFile(tmpfile, ext=0)

        # invalid ext number
        with pytest.raises(ValueError):
            f = FITSFile(tmpfile, ext=2)

@MPITest([1])
@pytest.mark.skipif(fitsio is None, "fitsio is not installed")
def test_no_tabular_data(comm):

    with temporary_data(data='image') as (data, tmpfile):

        # no binary table data
        with pytest.raises(ValueError):
            f = FITSFile(tmpfile)

        with pytest.raises(ValueError):
            f = FITSFile(tmpfile, ext=1)
