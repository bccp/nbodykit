from runtests.mpi import MPITest
from nbodykit.io.csv import CSVFile
import os
import numpy
import tempfile

@MPITest([1])
def test_dask(comm):

    with tempfile.NamedTemporaryFile() as ff:    
        
        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)
        
        # read
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=100)
        
        a_dask = f.get_dask('a')
        numpy.testing.assert_almost_equal(a_dask.compute(), data[:,0])

@MPITest([1])
def test_getitem(comm):

    with tempfile.NamedTemporaryFile() as ff:    
        
        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)
        
        # read
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=100)

        # empty slice
        try: empty = f[[]]
        except IndexError: pass
        
        # cannot slice twice
        a = f['a']
        try: a2 = a['a']
        except IndexError: pass
        
        # bad column name
        try: bad = f[['BAD1', 'BAD2']]
        except IndexError: pass
        
        # slice multiple columns
        f2 = f[['a', 'b']]
        assert f2.columns == ['a', 'b']
        f3 = f2[['a']]
        assert f3.columns == ['a']
        
        # slice as an array 
        d = f.asarray()
        numpy.testing.assert_almost_equal(d[:,0:2], data[:,0:2])
        numpy.testing.assert_almost_equal(d[:50], data[:50])
        numpy.testing.assert_almost_equal(d[0].squeeze(), data[0])
        
        # pass list of integers
        numpy.testing.assert_almost_equal(d[[0,1,2]], data[[0,1,2]])
        
        # pass boolean slice
        valid = numpy.random.choice([True, False], replace=True, size=len(f))
        f2 = f[valid]
        numpy.testing.assert_almost_equal(f2['a'][:], data[valid, 0])
        
        # wrong slice shape
        try: d2 = d[:,:,:]
        except IndexError: pass
        
@MPITest([1])
def test_asarray(comm):

    with tempfile.NamedTemporaryFile() as ff:    
        
        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)
        
        # read
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, dtype={'a':'f4', 'b':'f8'}, blocksize=100)
        
        # cannot do asarray twice
        a = f['a']
        try: a2 = a.asarray()
        except ValueError: pass
        
        # cannot do view with different dtypes
        try: f2 = f.asarray()
        except ValueError: pass
        
        
        
        