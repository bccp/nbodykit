from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose
import pytest

setup_logging()

@MPITest([1])
def test_missing_columns(comm):

    # create FKP catalog
    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)

    with pytest.raises(ValueError):
        cat = FKPCatalog(source1, source2, BoxSize=512.0, BoxPad=0.02)

@MPITest([1])
def test_boxsize(comm):

    # data and randoms
    source1 = UniformCatalog(nbar=3e-3, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-3, BoxSize=512., seed=84, comm=comm)

    # add required columns
    source1['NZ']  = 1.0
    source2['NZ']  = 1.0

    # create FKPCatalog
    cat = FKPCatalog(source1, source2, BoxPad=0.02)

    # no Nmesh?
    with pytest.raises(ValueError):
        mesh = cat.to_mesh()

    # mesh
    mesh = cat.to_mesh(Nmesh=32)

    # check boxsize
    assert_allclose(mesh.attrs['BoxSize'], numpy.ceil(1.02*512.))
