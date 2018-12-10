from runtests.mpi import MPITest
from nbodykit.source.catalog import SubVolumesCatalog
from numpy.testing import assert_allclose, assert_array_equal
from nbodykit import setup_logging
import pytest
from nbodykit.lab import *

setup_logging("debug")

@MPITest([1, 4])
def test_decomposed(comm):
    cosmo = cosmology.Planck15

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=1e-5, BoxSize=128., Nmesh=8, seed=42, comm=comm)

    decomposed = source.to_subvolumes()

    assert decomposed.comm == source.comm
    assert 'Position' in decomposed
    assert 'Velocity' in decomposed

    assert len(decomposed['Position'].compute()) == decomposed.size
    assert len(decomposed['Velocity'].compute()) == decomposed.size

    decomposed = source.to_subvolumes(columns=['Position'])
    assert decomposed.comm == source.comm

    assert 'Position' in decomposed
    assert 'Velocity' not in decomposed

    assert len(decomposed['Position'].compute()) == decomposed.size
