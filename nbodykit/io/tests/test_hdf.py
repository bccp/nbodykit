from runtests.mpi import MPITest
from nbodykit.io.hdf import HDFFile
import os
import numpy
import tempfile
import pickle
import contextlib

HAVE_H5PY = False
try: import h5py
except: HAVE_H5PY = True

@contextlib.contextmanager
def temporary_data():
    
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
            
        yield (dset, tmpfile)
    except:
        raise
    finally:
        os.unlink(tmpfile)
        
        
@MPITest([1])
def test_data(comm):

    with temporary_data() as (data, tmpfile):

        # read
        f = HDFFile(tmpfile)

        # check columns
        cols = ['X/Mass', 'X/Position', 'Y/Mass', 'Y/Position']
        assert(all(col in cols for col in f.columns))
        
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

@MPITest([1])
def test_nonzero_root(comm):
    
    with temporary_data() as (data, tmpfile):
        
        # read
        f = HDFFile(tmpfile, root='Y')
        
        # non-zero root
        f = HDFFile(tmpfile, root='Y')
        assert(all(col in ['Mass', 'Position'] for col in f.columns))
        
        # wrong root
        try: f = HDFFile(tmpfile, root='Z')
        except ValueError: pass


@MPITest([1])
def test_nonzero_exclude(comm):
    
    with temporary_data() as (data, tmpfile):
        
        # read
        f = HDFFile(tmpfile, exclude=['Y'])
        
        # non-zero exclude
        f = HDFFile(tmpfile, exclude=['Y'])
        assert(all(col in ['Mass', 'Position'] for col in f.columns))
        
        # non-zero exclude and root
        f = HDFFile(tmpfile, exclude=['Mass'], root='Y')
        assert all(col in ['Position'] for col in f.columns)
        
        # bad exclude
        try: f = HDFFile(tmpfile, exclude=['Z'])
        except ValueError: pass
        
@MPITest([1])
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
    try: f = HDFFile(tmpfile)
    except ValueError: pass
    
    # only one dataset now, so this works
    f = HDFFile(tmpfile, exclude=['Mass'])
    assert f.size == 1024

    os.unlink(tmpfile) 
        
@MPITest([1])
def test_empty(comm):
    
    # create empty file
    tmpfile = tempfile.mkstemp()[1]
    with h5py.File(tmpfile , 'w') as ff:
        ff.create_group('Y') 
    
    # no datasets!
    try: f = HDFFile(tmpfile)
    except ValueError: pass

    os.unlink(tmpfile) 
        