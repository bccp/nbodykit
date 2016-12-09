from mpi4py_test import MPIWorld
import numpy

@MPIWorld(NTask=[1])
def test_csv(comm):
    """
    Test reading data using ``CSVFile``
    """
    import tempfile
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
            numpy.testing.assert_allclose(data[:,i], f[names[i]][:], err_msg="error reading column '%s'" %names[i])

            
