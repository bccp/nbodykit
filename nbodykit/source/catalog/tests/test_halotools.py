from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
import shutil
import pytest

setup_logging()

@MPITest([4])
def test_cached_halotools(comm):

    from halotools.sim_manager import UserSuppliedHaloCatalog
    CurrentMPIComm.set(comm)

    # download and load the cached catalog
    cat = HalotoolsCachedCatalog('bolshoi', 'rockstar', 0.5)
    halotools_cat = cat.to_halotools()
    assert isinstance(halotools_cat, UserSuppliedHaloCatalog)

    # bad simulation name
    with pytest.raises(Exception):
        cat = HalotoolsCachedCatalog('BAD', 'rockstar', 0.5)


@MPITest([4])
def test_halotools_mock_catalog(comm):

    from halotools.empirical_models import PrebuiltHodModelFactory
    CurrentMPIComm.set(comm)

    # the cached catalog
    cat = HalotoolsCachedCatalog('bolshoi', 'rockstar', 0.5)
    halocat = cat.to_halotools() # this is a halotools UserSuppliedHaloCatalog

    # the zheng 07 model
    zheng07_model = PrebuiltHodModelFactory('zheng07', threshold = -19.5, redshift = 0.5)

    # make a mock
    mock = HalotoolsMockCatalog(halocat, zheng07_model)
    assert mock.csize > 0

    # delete a column so the mock population will fail
    del halocat.halo_table['halo_mvir']
    with pytest.raises(Exception):
        mock = HalotoolsMockCatalog(halocat, zheng07_model)
