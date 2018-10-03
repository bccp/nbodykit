from runtests.mpi import MPITest
from nbodykit.io.bigfile import BigFile
import shutil
import numpy
import tempfile
import pickle
import contextlib

@contextlib.contextmanager
def temporary_data():
    
    import bigfile
    try:
        data = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Velocity', ('f8',3))])
        data['Position'] = numpy.random.random(size=(1024, 3))
        data['Velocity'] = numpy.random.random(size=(1024,3))

        tmpdir = tempfile.mkdtemp()
        with bigfile.BigFile(tmpdir, create=True) as tmpff:
            with tmpff.create("Position", dtype=('f4', 3), size=1024) as bb:
                bb.write(0, data['Position'])
            with tmpff.create("Velocity", dtype=('f4', 3), size=1024) as bb:
                bb.write(0, data['Velocity'])
            with tmpff.create("Header") as bb:
                bb.attrs['Size'] = 1024
                bb.attrs['Over'] = 0

            with tmpff.create('1/.') as bb:
                bb.attrs['Size'] = 1024
                bb.attrs['Over'] = 1024

            with tmpff.create("1/Position", dtype=('f4', 3), size=1024) as bb:
                bb.write(0, data['Position'])
            with tmpff.create("1/Velocity", dtype=('f4', 3), size=1024) as bb:
                bb.write(0, data['Velocity'])

        yield (data, tmpdir)
    except:
        raise
    finally:
        shutil.rmtree(tmpdir)

@MPITest([1])
def test_data(comm):

    with temporary_data() as (data, tmpfile):
        # read
        ff = BigFile(tmpfile, header='Header', dataset='1')

        # check size
        assert ff.attrs['Size'] == 1024

        # and data
        numpy.testing.assert_almost_equal(data['Position'], ff['Position'][:])
        numpy.testing.assert_almost_equal(data['Velocity'], ff['Velocity'][:])

@MPITest([1])
def test_data_auto_header(comm):
    with temporary_data() as (data, tmpfile):
        ff = BigFile(tmpfile, dataset='1')

        # check size
        assert ff.attrs['Size'] == 1024
        assert ff.attrs['Over'] == 1024

@MPITest([1])
def test_pickle(comm):
    
    with temporary_data() as (data, tmpfile):
        
        # read
        ff = BigFile(tmpfile, header='Header', exclude=['1/*', 'Header'])
    
        # pickle
        s = pickle.dumps(ff)
        ff2 = pickle.loads(s)

        # check size
        assert ff2.attrs['Size'] == 1024
        
        # and data
        numpy.testing.assert_almost_equal(data['Position'], ff2['Position'][:])
        numpy.testing.assert_almost_equal(data['Velocity'], ff2['Velocity'][:])

