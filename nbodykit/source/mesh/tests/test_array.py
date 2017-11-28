from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose, assert_array_equal

setup_logging()

@MPITest([1, 4])
def test_paint(comm):

    from pmesh.pm import ParticleMesh

    # initialize a random white noise field in real space
    pm = ParticleMesh(Nmesh=(8, 8, 8), BoxSize=(128, 128, 128.), comm=comm)
    real = pm.generate_whitenoise(mode='real', seed=3333) # a RealField

    # FFT to a ComplexField
    complex = real.r2c()

    # gather to all ranks --> shape is now (8,8,8) on all ranks
    data = numpy.concatenate(comm.allgather(numpy.array(real.ravel())))\
                 .reshape(real.cshape)

    # mesh from data array
    realmesh = ArrayMesh(data, BoxSize=128., comm=comm)

    # paint() must yield the RealField
    assert_array_equal(real, realmesh.to_field(mode='real'))

    # gather complex to all ranks --> shape is (8,8,5) on all ranks
    cdata = numpy.concatenate(comm.allgather(numpy.array(complex.ravel())))\
                 .reshape(complex.cshape)

    # mesh from complex array
    cmesh = ArrayMesh(cdata, BoxSize=128., comm=comm)

    # paint() yields the ComplexField
    assert_allclose(complex, cmesh.to_field(mode='complex'), atol=1e-8)
