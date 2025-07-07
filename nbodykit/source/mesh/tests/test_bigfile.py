from nbodykit.lab import *
from nbodykit import setup_logging
from mpi4py import MPI
import shutil
from numpy.testing import assert_array_equal, assert_allclose
import pytest

setup_logging()

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_bigfile_grid(comm):

    import tempfile

    cosmo = cosmology.Planck15

    # input linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, BoxSize=512, Nmesh=32, seed=42, comm=comm)

    real = source.compute(mode='real')
    complex = source.compute(mode="complex")

    # and save to tmp directory
    if comm.rank == 0:
        output = tempfile.mkdtemp()
    else:
        output = None
    output = comm.bcast(output)

    source.save(output, dataset='Field')

    # now load it and paint to the algorithm's ParticleMesh
    source = BigFileMesh(path=output, dataset='Field', comm=comm)
    loaded_real = source.compute()

    # compare to direct algorithm result
    assert_array_equal(real, loaded_real)

    source.save(output, dataset='FieldC', mode='complex')

    # now load it and paint to the algorithm's ParticleMesh
    source = BigFileMesh(path=output, dataset='FieldC', comm=comm)
    loaded_real = source.compute(mode="complex")

    # compare to direct algorithm result
    assert_allclose(complex, loaded_real, atol=1e-7)
    if comm.rank == 0:
        shutil.rmtree(output)
