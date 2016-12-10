from mpi4py_test import MPIWorld
import numpy
import tempfile
import shutil
import pickle
from numpy.testing import assert_almost_equal

@MPIWorld(NTask=[1])
def test_csv(comm):
    """
    Test reading data using ``CSVFile``
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

def test_bigfile_pickle():
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

