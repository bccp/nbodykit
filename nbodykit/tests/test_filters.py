from nbodykit.lab import *
from nbodykit import setup_logging
from runtests.mpi import MPITest

import pytest

# debug logging
setup_logging("debug")

from nbodykit.filters import TopHat, Gaussian

@MPITest([1])
def test_tophat(comm):
    from nbodykit.base.mesh import MeshFilter

    cosmo = cosmology.Planck15

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42, comm=comm)
    source2 = source.apply(TopHat(8))

    r2 = source2.paint()
    # FIXME: add numerical assertions

@MPITest([1])
def test_tophat(comm):
    from nbodykit.base.mesh import MeshFilter

    cosmo = cosmology.Planck15

    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42, comm=comm)
    source2 = source.apply(Gaussian(8))
    r2 = source2.paint()
    # FIXME: add numerical assertions
