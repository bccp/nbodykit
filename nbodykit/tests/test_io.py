from mpi4py_test import MPITest
import numpy
import tempfile
import shutil
import pickle
import os
from numpy.testing import assert_almost_equal

@MPITest(1)
def test_csv(comm):
    """
    Test :class:`nbodykit.io.csv.CSVFile`
    """
    from nbodykit.io.csv import CSVFile
    
    with tempfile.NamedTemporaryFile() as ff:    
    
        # generate random data and write to temporary file
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data)
        ff.seek(0) # read from the beginning
        
        # read into a CSV file
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=1000)
        
        # check size
        numpy.testing.assert_equal(f.size, 100)
        
        # check values of each column
        for i, name in enumerate(names):
            assert_almost_equal(data[:,i], f[names[i]][:], err_msg="error reading column '%s'" %names[i])

        s = pickle.dumps(f)
        f2 = pickle.loads(s)

        # check values of each column
        for i, name in enumerate(names):
            assert_almost_equal(data[:,i], f2[names[i]][:], err_msg="error reading column '%s'" %names[i])

@MPITest(1)
def test_bigfile(comm):
    """
    Test :class:`nbodykit.io.bigfile.BigFile`
    """
    from nbodykit.io.bigfile import BigFile
    import bigfile
    tmpdir = tempfile.mkdtemp()

    pos = numpy.random.random(size=(1024, 3))
    vel = numpy.random.random(size=(1024, 3))

    with bigfile.BigFile(tmpdir, create=True) as tmpff:
        with tmpff.create("Position", dtype=('f4', 3), size=1024) as bb:
            bb.write(0, pos)
        with tmpff.create("Velocity", dtype=('f4', 3), size=1024) as bb:
            bb.write(0, vel)
        with tmpff.create("Header") as bb:
            bb.attrs['Size'] = 1024.

    ff = BigFile(tmpdir, header='Header')
    assert ff.attrs['Size'] == 1024
    assert_almost_equal(pos, ff.read(['Position'], 0, 1024)['Position'])
    assert_almost_equal(vel, ff.read(['Velocity'], 0, 1024)['Velocity'])

    # pickling?
    s = pickle.dumps(ff)
    ff2 = pickle.loads(s)

    assert ff2.attrs['Size'] == 1024
    assert_almost_equal(pos, ff2.read(['Position'], 0, 1024)['Position'])
    assert_almost_equal(vel, ff2.read(['Velocity'], 0, 1024)['Velocity'])
    shutil.rmtree(tmpdir)

@MPITest(1)
def test_binary(comm):
    """
    Test :class:`nbodykit.io.binary.BinaryFile`
    """
    from nbodykit.io.binary import BinaryFile

    pos = numpy.random.random(size=(1024, 3))
    vel = numpy.random.random(size=(1024, 3))

    tmpfile = tempfile.mkstemp()[1]

    with open(tmpfile , 'wb') as ff:
        pos.tofile(ff)
        vel.tofile(ff)

    f = BinaryFile(tmpfile, [('Position', ('f8', 3)), ('Velocity', ('f8', 3))], size=1024)

    numpy.testing.assert_equal(f.size, 1024)

    s = pickle.dumps(f)
    f2 = pickle.loads(s)

    assert_almost_equal(pos, f2.read(['Position'], 0, 1024)['Position'])
    assert_almost_equal(vel, f2.read(['Velocity'], 0, 1024)['Velocity'])

    os.unlink(tmpfile)
    
@MPITest(1)
def test_hdf(comm):
    """
    Test :class:`nbodykit.io.hdf.HDFFile`
    """
    from nbodykit.io.hdf import HDFFile
    import h5py

    # fake structured array
    dset = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Mass', 'f8')])
    dset['Position'] = numpy.random.random(size=(1024, 3))
    dset['Mass'] = numpy.random.random(size=1024)

    tmpfile = tempfile.mkstemp()[1]
    
    with h5py.File(tmpfile , 'w') as ff:
        ff.create_dataset('X', data=dset) # store structured array as dataset
        grp = ff.create_group('Y')
        grp.create_dataset('Position', data=dset['Position']) # column as dataset
        grp.create_dataset('Mass', data=dset['Mass']) # column as dataset

    f = HDFFile(tmpfile)
    
    # check columns
    cols = ['X/Mass', 'X/Position', 'Y/Mass', 'Y/Position']
    assert(all(col in cols for col in f.columns))
    for col in cols:
        field = col.rsplit('/', 1)[-1]
        assert_almost_equal(dset[field], f[col][:])
    
    # check size
    numpy.testing.assert_equal(f.size, 1024)

    # pickle test
    s = pickle.dumps(f)
    f2 = pickle.loads(s)
    for col in cols:
        field = col.rsplit('/', 1)[-1]
        assert_almost_equal(dset[field], f2[col][:])
    
    # non-zero root
    f = HDFFile(tmpfile, root='Y')
    cols = ['Mass', 'Position']
    assert(all(col in cols for col in f.columns))
    
    # non-zero exclude
    f = HDFFile(tmpfile, exclude=['Y'])
    cols = ['Mass', 'Position']
    assert(all(col in cols for col in f.columns))
    
    os.unlink(tmpfile)
