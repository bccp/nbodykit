from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose, assert_array_equal
import pytest

setup_logging("debug")

@MPITest([1, 4])
def test_table(comm):

    from astropy.table import Table

    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )
    data = Table(data)
    source = ArrayCatalog(data, BoxSize=100, Nmesh=32, comm=comm)

    for col in ['Position', 'Velocity']:
        assert_array_equal(data[col], source[col])


@MPITest([1, 4])
def test_nonstructured_input(comm):

    from astropy.table import Table

    # data should be structured!
    data = numpy.ones(100)
    with pytest.raises(ValueError):
        source = ArrayCatalog(data, BoxSize=100, Nmesh=32, comm=comm)




@MPITest([1, 4])
def test_array(comm):

    cosmo = cosmology.Planck15

    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )
    source = ArrayCatalog(data, BoxSize=100, Nmesh=32, comm=comm)

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

    data = numpy.ones(100, dtype=[
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3))]
            )
    # use a dictionary
    data = dict(Position=data['Position'], Velocity=data['Velocity'])
    source = ArrayCatalog(data, BoxSize=100, Nmesh=32, comm=comm)

    assert source.csize == 100 * comm.size
    source['Velocity'] = source['Position'] + source['Velocity']
    source['Position'] = source['Position'] + source['Velocity']
    assert_allclose(source['Position'], 3)
