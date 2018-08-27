from runtests.mpi import MPITest
from nbodykit.tutorials import DemoHaloCatalog
from nbodykit import setup_logging
import pytest

setup_logging()

@MPITest([4])
def test_download(comm):

    from halotools.sim_manager import UserSuppliedHaloCatalog

    # download and load the cached catalog
    cat = DemoHaloCatalog('bolshoi', 'rockstar', 0.5, comm=comm)
    assert all(col in cat for col in ['Position', 'Velocity'])

    # convert to halotools catalog
    halotools_cat = cat.to_halotools()
    assert isinstance(halotools_cat, UserSuppliedHaloCatalog)

    # bad simulation name
    with pytest.raises(Exception):
        cat = DemoHaloCatalog('BAD', 'rockstar', 0.5)


@MPITest([4])
def test_download_failure(comm):
    # initialize with bad redshift
    BAD_REDSHIFT = 100.0
    with pytest.raises(Exception):
        cat = DemoHaloCatalog('bolshoi', 'rockstar', BAD_REDSHIFT, comm=comm)
