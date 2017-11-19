from runtests.mpi import MPITest
from nbodykit.tutorials import download_example_data
from nbodykit import setup_logging, CurrentMPIComm
from six.moves.urllib.error import HTTPError
import pytest
import shutil
import os
import tempfile

setup_logging()

@pytest.mark.xfail(raises=HTTPError)
@MPITest([1])
def test_download_directory(comm):

    # download Gadget1P snapshots directory
    filename = 'Gadget1P'
    download_example_data(filename)

    # check that it worked
    assert filename in os.listdir('.')
    assert len(os.listdir(filename)) > 0

    # remove the downloaded file
    shutil.rmtree(filename)

@pytest.mark.xfail(raises=HTTPError)
@MPITest([1])
def test_download_file(comm):

    # download TPM snapshot file
    filename = 'tpm_1.0000.bin.00'
    download_example_data(filename)

    # check that it worked
    assert filename in os.listdir('.')

    # remove filename
    os.remove(filename)

@pytest.mark.xfail(raises=HTTPError)
@MPITest([1])
def test_download_failure(comm):

    filename = 'MISSING'
    with pytest.raises(ValueError):
        download_example_data(filename)

@pytest.mark.xfail(raises=HTTPError)
@MPITest([1])
def test_missing_dirname(comm):

    with pytest.raises(ValueError):
        download_example_data('Gadget1P', download_dirname='MISSING')

@pytest.mark.xfail(raises=HTTPError)
@MPITest([1])
def test_download_to_location(comm):

    # download Gadget1P snapshots directory to specific directory
    filename = 'Gadget1P'
    loc = tempfile.mkdtemp()
    download_example_data(filename, download_dirname=loc)

    # check that it worked
    assert filename in os.listdir(loc)

    # delete temp directory
    shutil.rmtree(loc)
