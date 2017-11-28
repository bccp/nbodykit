from runtests.mpi import MPITest
from nbodykit.io.csv import CSVFile
import os
import numpy
import tempfile
import pickle
import pytest


@MPITest([1])
def test_no_trailing_newline(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data with blank lines
        data = numpy.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype='i4')
        ff.write(("1 1 1 1\n2 2 2 2").encode())
        ff.seek(0)

        # this will fail -- header should not be in file
        names =['a', 'b', 'c', 'd']
        f = CSVFile(path=ff.name, names=names, blocksize=1000)

        assert(f.size == 2)
        data2 = f.asarray()
        numpy.testing.assert_almost_equal(data, data2[:], decimal=7)

@MPITest([1])
def test_data(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # read nrows
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=100)

        # make sure data is the same
        data2 = f.asarray()
        numpy.testing.assert_almost_equal(data, data2[:], decimal=7)

        # make sure all the columns are there
        assert all(col in f for col in names)

@MPITest([1])
def test_pickle(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # read nrows
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=100)

        s = pickle.dumps(f)
        f2 = pickle.loads(s)

        numpy.testing.assert_almost_equal(f['a'][:], f2['a'][:])

@MPITest([1])
def test_slicing(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # read nrows
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=1000)

        # make sure data is the same (check only the first column her)
        for sl in [slice(0,10), slice(-10, -5), slice(0, 50, 2), slice(-50, None, 3)]:
            numpy.testing.assert_almost_equal(data[sl][:,0], f[sl]['a'], decimal=7)

@MPITest([1])
def test_comma_sep(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e', delimiter=','); ff.seek(0)

        # this must fail
        names =['a', 'b', 'c', 'd', 'e']
        with pytest.raises(ValueError):
            f = CSVFile(path=ff.name, names=names, blocksize=100, delim_whitespace=True)

        # use , as delimiter
        f = CSVFile(path=ff.name, names=names, blocksize=100, delim_whitespace=False, sep=',')
        data2 = f.asarray()
        numpy.testing.assert_almost_equal(data, data2[:], decimal=7)


@MPITest([1])
def test_header_fail(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data with blank lines
        data = numpy.random.random(size=(100,5))
        ff.write(("# a b c d e\n").encode())
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # this will fail -- header should not be in file
        names =['a', 'b', 'c', 'd', 'e']
        with pytest.raises(ValueError):
            f = CSVFile(path=ff.name, names=names, blocksize=1000)


@MPITest([1])
def test_blank_lines(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data with blank lines
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # read
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=100)

        # the right size
        assert f.size == len(f[:]), "mismatch between 'size' and data read from file"
        assert f.size == 100, "error reading with blank lines'"

@MPITest([1])
def test_dtype(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        ff.write(("\n\n\n").encode())
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # specify the dtype as dict
        names =['a', 'b', 'c', 'd', 'e']
        dtype = {'a':'f4', 'b':'i8', 'c':'f16'}
        f = CSVFile(path=ff.name, names=names, blocksize=100, dtype=dtype)

        # make sure data is the same
        assert f.dtype['a'] == 'f4'
        assert f.dtype['b'] == 'i8'
        assert f.dtype['c'] == 'f16'

        # specify the dtype as dict
        f = CSVFile(path=ff.name, names=names, blocksize=100, dtype="f4")

        # make sure data is the same
        assert all(f.dtype[col] == 'f4' for col in names)


@MPITest([1])
def test_comments(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        ff.write(("# comment line 1\n# comment line 2\n").encode())
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # this should raise an error
        names =['a', 'b', 'c', 'd', 'e']
        with pytest.raises(ValueError):
            f = CSVFile(path=ff.name, names=names, blocksize=1000)


@MPITest([1])
def test_skiprows(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data, fmt='%.7e'); ff.seek(0)

        # read nrows
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=1000, nrows=50, skiprows=25)

        # the right size
        assert f.size == len(f[:]), "mismatch between 'size' and data read from file"
        assert f.size == 50, "error combining 'skiprows' and 'nrows'"

        # make sure right portion of data was read
        data2 = f.asarray()
        numpy.testing.assert_almost_equal(data[25:75], data2[:], decimal=7)

@MPITest([1])
def test_nrows(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data); ff.seek(0)

        # read nrows
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=1000, nrows=50)

        # the right size
        assert f.size == len(f[:]), "mismatch between 'size' and data read from file"
        assert f.size == 50, "error reading 'nrows'"

        # make sure right portion of data was read
        data2 = f.asarray()
        numpy.testing.assert_almost_equal(data[:50], data2[:], decimal=7)

@MPITest([1])
def test_usecols(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data); ff.seek(0)

        # read usecols
        names =['a', 'b', 'c', 'd', 'e']
        f = CSVFile(path=ff.name, names=names, blocksize=1000, usecols=['a', 'c', 'e'])

        assert f.columns == ['a', 'c', 'e'], "error using 'usecols'"

@MPITest([1])
def test_wrong_names(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data); ff.seek(0)

        # wrong number of columns
        names =['a', 'b', 'c']
        with pytest.raises(ValueError):
            f = CSVFile(path=ff.name, names=names, blocksize=1000)


@MPITest([1])
def test_invalid_keywords(comm):

    with tempfile.NamedTemporaryFile() as ff:

        # generate data
        data = numpy.random.random(size=(100,5))
        numpy.savetxt(ff, data); ff.seek(0)

        # the bad keywords
        bad_kws = {'index_col':True, 'header':True, 'skiprows':[0,1,2], 'skipfooter':True, 'comment':'#'}

        # try each bad kewyord
        names =['a', 'b', 'c', 'd', 'e']
        for k,v in bad_kws.items():
            with pytest.raises(ValueError):
                f = CSVFile(path=ff.name, names=names, blocksize=1000, **{k:v})
