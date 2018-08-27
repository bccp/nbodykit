from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from nbodykit.utils import ScatterArray, GatherArray, FrontPadArray
from numpy.testing import assert_array_equal
import os
import pytest

setup_logging("debug")

@MPITest([1])
def test_json_quantity(comm):

    from nbodykit.utils import JSONEncoder, JSONDecoder
    import json
    import tempfile

    cosmo = cosmology.Planck15

    # astropy quantity
    cosmo2 = cosmo.to_astropy()
    pos = numpy.ones(10, dtype=[('Position', ('f4',3))])
    data = {'m_nu':cosmo2.m_nu, 'H0':cosmo2.H0, 'a':numpy.float64(10.0), 'b':10, 'pos':pos}

    # write to file
    tmpfile = tempfile.mktemp()
    with open(tmpfile, 'w') as ff:
        json.dump(data, ff, cls=JSONEncoder)

    # load from file
    with open(tmpfile, 'r') as ff:
        data2 = json.load(ff, cls=JSONDecoder)

    # test for equality
    for name in ['H0', 'm_nu']:
        assert data2[name].__class__ == data[name].__class__
        numpy.testing.assert_array_equal(data2[name], data[name])
    assert data['a'] == data2['a']
    assert data['b'] == data2['b']

    os.remove(tmpfile)

@MPITest([2])
def test_gather_array(comm):

    # object arrays must fail
    data1a = numpy.ones(10, dtype=[('test', 'f8')])
    data2a = numpy.ones(10, dtype='f8')

    data1 = GatherArray(data1a, comm, root=0)
    if comm.rank == 0:
        numpy.testing.assert_array_equal(data1['test'], 1)
        assert len(data1) == 10 * comm.size
    else:
        assert data1 is None

    data2 = GatherArray(data2a, comm, root=0)
    if comm.rank == 0:
        numpy.testing.assert_array_equal(data2, 1)
        assert len(data2) == 10 * comm.size
    else:
        assert data2 is None

    data2 = GatherArray(data2a, comm, root=Ellipsis)
    numpy.testing.assert_array_equal(data2, 1)
    assert len(data2) == 10 * comm.size

    data1 = GatherArray(data1a, comm, root=Ellipsis)
    numpy.testing.assert_array_equal(data1['test'], 1)
    assert len(data1) == 10 * comm.size


@MPITest([2])
def test_gather_objects(comm):

    # object arrays must fail
    data1 = numpy.ones(10, dtype=[('test', 'O')])
    data2 = numpy.ones(10, dtype='O')

    with pytest.raises(ValueError):
        data1 = GatherArray(data1, comm, root=0)

    with pytest.raises(ValueError):
        data2 = GatherArray(data2, comm, root=0)

    with pytest.raises(ValueError):
        data1 = GatherArray(data1, comm, root=Ellipsis)

    with pytest.raises(ValueError):
        data2 = GatherArray(data2, comm, root=Ellipsis)

@MPITest([2])
def test_scatter_objects(comm):

    # object arrays must fail
    if comm.rank == 0:
        data1 = numpy.ones(10, dtype=[('test', 'O')])
        data2 = numpy.ones(10, dtype='O')
    else:
        data1 = None
        data2 = None

    with pytest.raises(ValueError):
        data1 = ScatterArray(data1, comm, root=0)

    with pytest.raises(ValueError):
        data2 = ScatterArray(data2, comm, root=0)

@MPITest([2])
def test_gather_bad_data(comm):

    # data
    if comm.rank == 0:
        data = numpy.ones(10, dtype=[('a', 'f')])
    else:
        data = numpy.ones(10, dtype=[('b', 'f')])

    # fields mismatch
    with pytest.raises(ValueError):
        data = GatherArray(data, comm, root=0)

    with pytest.raises(ValueError):
        data = GatherArray(data, comm, root=Ellipsis)

@MPITest([2])
def test_gather_bad_dtype(comm):

    # data
    if comm.rank == 0:
        data = numpy.ones(10, dtype=[('a', 'f4')])
    else:
        data = numpy.ones(10, dtype=[('a', 'f8')])

    # shape mismatch
    with pytest.raises(ValueError):
        data = GatherArray(data, comm, root=0)

    with pytest.raises(ValueError):
        data = GatherArray(data, comm, root=Ellipsis)

@MPITest([2])
def test_gather_bad_shape(comm):

    # data
    if comm.rank == 0:
        data = numpy.ones((10,2))
    else:
        data = numpy.ones((10,3))

    # shape mismatch
    with pytest.raises(ValueError):
        data = GatherArray(data, comm, root=0)

    with pytest.raises(ValueError):
        data = GatherArray(data, comm, root=Ellipsis)

@MPITest([2])
def test_gather_list(comm):

    # data
    data = numpy.ones(10, dtype=[('a', 'f')])

    # can't gather a list
    with pytest.raises(ValueError):
        data = GatherArray(list(data), comm, root=0)

    with pytest.raises(ValueError):
        data = GatherArray(list(data), comm, root=Ellipsis)

@MPITest([2])
def test_scatter_list(comm):

    # data
    if comm.rank == 0:
        data = list(numpy.ones(10, dtype=[('a', 'f')]))
    else:
        data = None

    # can't scatter list
    with pytest.raises(ValueError):
        data = ScatterArray(data, comm, root=0)


@MPITest([2])
def test_scatter_wrong_counts(comm):

    # data
    if comm.rank == 0:
        data = numpy.ones(10, dtype=[('a', 'f')])
    else:
        data = None

    # wrong counts length
    with pytest.raises(ValueError):
        data = ScatterArray(data, comm, root=0, counts=[0, 5, 5])

    # wrong counts sum
    with pytest.raises(ValueError):
        data = ScatterArray(data, comm, root=0, counts=[5, 7])

@MPITest([4])
def test_frontpad_array(comm):

    # object arrays must fail
    data1 = numpy.ones(10) * comm.rank

    data2 = FrontPadArray(data1, comm.rank * 2, comm)
    assert len(data2) == len(data1) + comm.rank * 2
    assert_array_equal(data2[:comm.rank * 2], comm.rank - 1)
    assert_array_equal(data2[comm.rank * 2:], data1)

    data2 = FrontPadArray(data1, comm.rank * 10, comm)
    assert len(data2) == len(data1) + comm.rank * 10

    assert_array_equal(data2[comm.rank * 10:], data1)

    for i in range(comm.rank):
        assert_array_equal(data2[(comm.rank - i - 1) * 10:(comm.rank - i)* 10], comm.rank - i - 1)
