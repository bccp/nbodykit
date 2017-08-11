from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose, assert_array_equal
import os

setup_logging("debug")
data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')

@MPITest([4])
def test_threeptcf(comm):

    import tempfile

    CurrentMPIComm.set(comm)
    BoxSize = 400.0

    # load the test data
    filename = os.path.join(data_dir, 'threeptcf_data.dat')
    cat = CSVCatalog(filename, names=['x', 'y', 'z', 'w'])
    cat['Position'] = transform.StackColumns(cat['x'], cat['y'], cat['z'])
    cat['Position'] *= BoxSize

    # r binning
    nbins = 8
    edges = numpy.linspace(0, 200.0, nbins+1)

    # run the algorithm
    ells = list(range(0, 11))
    r = Multipoles3PCF(cat, ells, edges, BoxSize=BoxSize, weight='w')

    # load the result from file
    truth = numpy.empty((8,8,11))
    with open(os.path.join(data_dir, 'threeptcf_result.dat'), 'r') as ff:
        for line in ff:
            fields = line.split()
            i, j = int(fields[0]), int(fields[1])
            truth[i,j,:] = list(map(float, fields[2:]))
            truth[j,i,:] = truth[i,j,:]

    # test equality
    for i, ell in enumerate(ells):
        x = r.poles['zeta_%d' %ell]
        assert_allclose(x * (4*numpy.pi)**2 / (2*ell+1), truth[...,i], rtol=1e-3, err_msg='mismatch for ell=%d' %ell)

    # save to temp file
    tmpfile = tempfile.mktemp()
    r.save(tmpfile)

    r2 = Multipoles3PCF.load(tmpfile)
    assert_array_equal(r.poles.data, r2.poles.data)
