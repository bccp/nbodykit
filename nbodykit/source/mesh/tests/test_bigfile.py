from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

import shutil
from numpy.testing import assert_array_equal, assert_allclose

setup_logging()

@MPITest([1,4])
def test_bigfile_grid(comm):

    import tempfile

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # input linear mesh
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, BoxSize=512, Nmesh=64, seed=42)

    real = source.paint(mode='real')
    complex = source.paint(mode="complex")

    # and save to tmp directory
    if comm.rank == 0:
        output = tempfile.mkdtemp()
    else:
        output = None
    output = comm.bcast(output)

    source.save(output, dataset='Field')

    # now load it and paint to the algorithm's ParticleMesh
    source = BigFileMesh(path=output, dataset='Field')
    loaded_real = source.paint()

    # compare to direct algorithm result
    assert_array_equal(real, loaded_real)

    source.save(output, dataset='FieldC', mode='complex')

    # now load it and paint to the algorithm's ParticleMesh
    source = BigFileMesh(path=output, dataset='FieldC')
    loaded_real = source.paint(mode="complex")

    # compare to direct algorithm result
    assert_allclose(complex, loaded_real, atol=1e-7)
    if comm.rank == 0:
        shutil.rmtree(output)
