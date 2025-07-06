from nbodykit.lab import *
from nbodykit import setup_logging
import pytest
from mpi4py import MPI

setup_logging()

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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
