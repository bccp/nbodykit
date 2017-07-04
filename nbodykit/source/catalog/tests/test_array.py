from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose

setup_logging("debug")

@MPITest([1, 4])
def test_array(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)
    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )
    source = ArrayCatalog(data, BoxSize=100, Nmesh=32)

    assert source.csize == 100 * comm.size
    source['Velocity'] = source['Position'] + source['Velocity']
    source['Position'] = source['Position'] + source['Velocity']

    # Position triggers  Velocity which triggers Position and Velocity
    # which resolves to the true data.
    # so total is 3.
    assert_allclose(source['Position'], 3)


@MPITest([1, 4])
def test_dict(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)
    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )
    # use a dictionary
    data = dict(Position=data['Position'], Velocity=data['Velocity'])
    source = ArrayCatalog(data, BoxSize=100, Nmesh=32)

    assert source.csize == 100 * comm.size
    source['Velocity'] = source['Position'] + source['Velocity']
    source['Position'] = source['Position'] + source['Velocity']
    assert_allclose(source['Position'], 3)
