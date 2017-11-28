from runtests.mpi import MPITest
from nbodykit.io.csv import CSVFile

import os
import numpy
import tempfile
import pytest

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
        with pytest.raises(IndexError):
            empty = f[[]]

        # cannot slice twice
        a = f['a']
        with pytest.raises(IndexError):
            a2 = a['a']
        
        # bad column name
        with pytest.raises(IndexError):
            bad = f[['BAD1', 'BAD2']]
        
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
        with pytest.raises(IndexError):
            d2 = d[:,:,:]

        
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
        with pytest.raises(ValueError):
            a2 = a.asarray()
        
        # cannot do view with different dtypes
        with pytest.raises(ValueError):
            f2 = f.asarray()
