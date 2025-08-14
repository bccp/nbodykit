from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose, assert_array_equal
from mpi4py import MPI
import pytest

setup_logging()

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_paint(comm):
    from pmesh.pm import ParticleMesh

    pm = ParticleMesh(Nmesh=(8, 8, 8), BoxSize=(128, 128, 128.), comm=comm)
    real = pm.generate_whitenoise(type='real', seed=3333)
    complex = real.r2c()

    realmesh = FieldMesh(real)
    complexmesh = FieldMesh(complex)

    assert_array_equal(real, realmesh.to_field(mode='real'))
    assert_array_equal(complex, complexmesh.to_field(mode='complex'))

    assert_allclose(complex, realmesh.to_field(mode='complex'))
    assert_allclose(real, complexmesh.to_field(mode='real'))
