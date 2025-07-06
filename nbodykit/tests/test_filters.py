from nbodykit import cosmology
from nbodykit import setup_logging
import pytest
from nbodykit.base.mesh import MeshFilter
from nbodykit.filters import TopHat
from nbodykit.source.mesh import LinearMesh
from mpi4py import MPI

# debug logging
setup_logging("debug")

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_tophat(comm):
    cosmo = cosmology.Planck15

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42, comm=comm)
    source2 = source.apply(TopHat(8))

    r2 = source2.paint()
    # FIXME: add numerical assertions
