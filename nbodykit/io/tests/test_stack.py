from runtests.mpi import MPITest
from nbodykit.io.tpm import TPMBinaryFile
from nbodykit.io.stack import FileStack
import numpy
import tempfile
import os
import shutil
import contextlib
import pytest

@contextlib.contextmanager
def TemporaryDirectory():
    try:
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


@MPITest([1])
def test_data(comm):

    with TemporaryDirectory() as tmpdir:
        
        # generate TPM-format data
        pos = numpy.random.random(size=(2048, 3)).astype('f4')
        vel = numpy.random.random(size=(2048, 3)).astype('f4')
        uid = numpy.arange(2048, dtype='u8')
        hdr = numpy.ones(28, dtype='?')
        
        for i, name in enumerate(['tpm.000', 'tpm.001']):
            sl = slice(i*1024, (i+1)*1024)
            
            # write to file
            fname = os.path.join(tmpdir, name)
            with open(fname, 'wb') as ff:
                hdr.tofile(ff)
                pos[sl].tofile(ff); vel[sl].tofile(ff); uid[sl].tofile(ff)
            
        # initialize the stack
        path = os.path.join(tmpdir, 'tpm.00*')
        f = FileStack(TPMBinaryFile, path, precision='f4')
        
        # check size
        assert f.size == 2048
        
        # and data
        numpy.testing.assert_almost_equal(pos, f['Position'][:])
        numpy.testing.assert_almost_equal(vel, f['Velocity'][:])
        numpy.testing.assert_almost_equal(uid, f['ID'][:])
        
        # pass a list
        paths = [os.path.join(tmpdir, f) for f in ['tpm.000', 'tpm.001']]
        f = FileStack(TPMBinaryFile, paths, precision='f4')
        
        # check size
        assert f.size == 2048
        assert f.nfiles == 2
        
        # and add and attrs
        f.attrs['size'] = 2048
        

@MPITest([1])
def test_single_path(comm):

    with TemporaryDirectory() as tmpdir:
        
        # generate TPM-format data
        pos = numpy.random.random(size=(2048, 3)).astype('f4')
        vel = numpy.random.random(size=(2048, 3)).astype('f4')
        uid = numpy.arange(2048, dtype='u8')
        hdr = numpy.ones(28, dtype='?')
        
        for i, name in enumerate(['tpm.000', 'tpm.001']):
            sl = slice(i*1024, (i+1)*1024)
            
            # write to file
            fname = os.path.join(tmpdir, name)
            with open(fname, 'wb') as ff:
                hdr.tofile(ff)
                pos[sl].tofile(ff); vel[sl].tofile(ff); uid[sl].tofile(ff)

        # single path
        f = FileStack(TPMBinaryFile, os.path.join(tmpdir, 'tpm.000'), precision='f4')
        assert f.size == 1024
        assert f.nfiles == 1


@MPITest([1])
def test_bad_path(comm):

    with TemporaryDirectory() as tmpdir:
        
        # generate TPM-format data
        pos = numpy.random.random(size=(2048, 3)).astype('f4')
        vel = numpy.random.random(size=(2048, 3)).astype('f4')
        uid = numpy.arange(2048, dtype='u8')
        hdr = numpy.ones(28, dtype='?')
        
        for i, name in enumerate(['tpm.000', 'tpm.001']):
            sl = slice(i*1024, (i+1)*1024)
            
            # write to file
            fname = os.path.join(tmpdir, name)
            with open(fname, 'wb') as ff:
                hdr.tofile(ff)
                pos[sl].tofile(ff); vel[sl].tofile(ff); uid[sl].tofile(ff)
            
        # bad path name
        with pytest.raises(ValueError): 
            f = FileStack(TPMBinaryFile, ff, precision='f4')
