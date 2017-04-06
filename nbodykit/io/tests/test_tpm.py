from runtests.mpi import MPITest
from nbodykit.io.tpm import TPMBinaryFile
import numpy
import tempfile
import os
import pytest

@MPITest([1])
def test_data(comm):

    tmpfile = tempfile.mktemp()
    with open(tmpfile, 'wb') as ff: 
        
        # generate TPM-format data
        pos = numpy.random.random(size=(1024, 3)).astype('f4')
        vel = numpy.random.random(size=(1024, 3)).astype('f4')
        uid = numpy.arange(1024, dtype='u8')
        hdr = numpy.ones(28, dtype='?')
        
        # write to file
        hdr.tofile(ff)
        pos.tofile(ff); vel.tofile(ff); uid.tofile(ff)
        ff.seek(0)
        
        # read
        f = TPMBinaryFile(ff.name, precision='f4')
        
        # check size
        assert f.size == 1024
        
        # and data
        numpy.testing.assert_almost_equal(pos, f['Position'][:])
        numpy.testing.assert_almost_equal(vel, f['Velocity'][:])
        numpy.testing.assert_almost_equal(uid, f['ID'][:])
        
        # check wrong precision
        with pytest.raises(ValueError): 
            f = TPMBinaryFile(ff.name, precision='f16')
        
    os.remove(tmpfile)