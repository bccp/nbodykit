from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

setup_logging()

@MPITest([4])
def test_bad_init(comm):

    # initialize a catalog
    cat = UniformCatalog(nbar=100, BoxSize=1.0, comm=comm)
    cat['Mass'] = 1.0

    # cannot specify column as None
    with pytest.raises(ValueError):
        halos = HaloCatalog(cat, cosmology.Planck15, 0., mass=None)

    # missing column
    with pytest.raises(ValueError):
        halos = HaloCatalog(cat, cosmology.Planck15, 0., mass='MISSING')

@MPITest([4])
def test_missing_boxsize(comm):

    # initialize a catalog
    cat = UniformCatalog(nbar=100, BoxSize=1.0, comm=comm)
    cat['Mass'] = 1.0

    # initialize halos
    halos = HaloCatalog(cat, cosmology.Planck15, 0.)

    # delete BoxSize
    del halos.attrs['BoxSize']

    # missing BoxSize!
    with pytest.raises(ValueError):
        halocat = halos.to_halotools()
