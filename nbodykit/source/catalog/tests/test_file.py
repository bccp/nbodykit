from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose

@MPITest([1])
def test_file(comm):

    import h5py
    import tempfile
    from nbodykit.io.hdf import HDFFile
    import os

    # fake structured array
    dset = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Mass', 'f8')])
    dset['Position'] = numpy.random.random(size=(1024, 3))
    dset['Mass'] = numpy.random.random(size=1024)

    tmpfile = tempfile.mkstemp()[1]

    with h5py.File(tmpfile , 'w') as ff:
        ds = ff.create_dataset('X', data=dset) # store structured array as dataset
        ds.attrs['BoxSize'] = 1.0
        grp = ff.create_group('Y')
        grp.create_dataset('Position', data=dset['Position']) # column as dataset
        grp.create_dataset('Mass', data=dset['Mass']) # column as dataset

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    source = HDFCatalog(tmpfile, root='X', attrs={"Nmesh":32})
    assert_allclose(source['Position'], dset['Position'])

    os.unlink(tmpfile)
