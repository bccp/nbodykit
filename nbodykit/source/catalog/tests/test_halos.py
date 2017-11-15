from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
import shutil
import pytest

setup_logging()

@MPITest([4])
def test_bad_init(comm):

    CurrentMPIComm.set(comm)

    # initialize a catalog
    cat = UniformCatalog(nbar=100, BoxSize=1.0)
    cat['Mass'] = 1.0

    # cannot specify column as None
    with pytest.raises(ValueError):
        halos = HaloCatalog(cat, mass=None)

    # missing column
    with pytest.raises(ValueError):
        halos = HaloCatalog(cat, mass='MISSING')

@MPITest([4])
def test_missing_boxsize(comm):

    CurrentMPIComm.set(comm)

    # initialize a catalog
    cat = UniformCatalog(nbar=100, BoxSize=1.0)
    cat['Mass'] = 1.0

    # initialize halos
    halos = HaloCatalog(cat, mass=None)

    # delete BoxSize
    del halos.attrs['BoxSize']

    # missing BoxSize!
    with pytest.raises(ValueError):
        halocat = halos.to_halotools()


@MPITest([4])
def test_demo_halos(comm):

    from halotools.sim_manager import UserSuppliedHaloCatalog
    CurrentMPIComm.set(comm)

    # download and load the cached catalog
    cat = DemoHaloCatalog('bolshoi', 'rockstar', 0.5)
    assert all(col in cat for col in ['Position', 'Velocity'])

    # convert to halotools catalog
    halotools_cat = cat.to_halotools()
    assert isinstance(halotools_cat, UserSuppliedHaloCatalog)

    # bad simulation name
    with pytest.raises(Exception):
        cat = DemoHaloCatalog('BAD', 'rockstar', 0.5)


@MPITest([4])
def test_demo_halos(comm):
    CurrentMPIComm.set(comm)

    # initialize with bad redshift
    BAD_REDSHIFT = 100.0
    with pytest.raises(Exception):
        cat = DemoHaloCatalog('bolshoi', 'rockstar', BAD_REDSHIFT)
