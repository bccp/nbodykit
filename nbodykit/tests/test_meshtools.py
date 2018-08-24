from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from nbodykit.meshtools import SlabIterator

from pmesh.pm import ParticleMesh, RealField, ComplexField
import pytest
from numpy.testing import assert_array_equal

# debug logging
setup_logging("debug")

@MPITest([1])
def test_wrong_ndim(comm):

    numpy.random.seed(42)

    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8,8], comm=comm, dtype='f8')
    rfield = RealField(pm)
    data = numpy.random.random(size=rfield.shape)
    rfield[...] = data[:]

    # SlabIterator only works for 2D or 3D coordinate meshes
    with pytest.raises(NotImplementedError):
        for slab in SlabIterator([rfield.x[0]], axis=0, symmetry_axis=None):
            pass

@MPITest([1])
def test_wrong_coords_shape(comm):

    numpy.random.seed(42)

    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    rfield = RealField(pm)
    data = numpy.random.random(size=rfield.shape)
    rfield[...] = data[:]

    # coords arrays should not be squeezed
    x = [numpy.squeeze(xx) for xx in rfield.x]

    with pytest.raises(ValueError):
        for slab in SlabIterator(x, axis=0, symmetry_axis=None):
            pass


@MPITest([1, 4])
def test_2d_slab(comm):

    numpy.random.seed(42)

    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    rfield = RealField(pm)
    data = numpy.random.random(size=rfield.shape)
    rfield[...] = data[:]

    x = rfield.x
    for i, slab in enumerate(SlabIterator(x, axis=0, symmetry_axis=None)):
        assert slab.__str__() == slab.__repr__()
        assert slab.shape == (8,)
        assert_array_equal(slab.hermitian_weights, numpy.ones(slab.shape))
        assert_array_equal(rfield[slab.index],  data[i])

@MPITest([1, 4])
def test_hermitian_weights(comm):

    numpy.random.seed(42)

    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8, 8], comm=comm, dtype='f8')
    cfield = ComplexField(pm)
    data = numpy.random.random(size=cfield.shape)
    data = data[:] + 1j*data[:]

    cfield[...] = data[:]
    x = cfield.x

    # iterate over symmetry axis
    for i, slab in enumerate(SlabIterator(x, axis=2, symmetry_axis=2)):

        # nonsingular weights give indices of positive frequencies
        nonsig = slab.nonsingular
        weights = slab.hermitian_weights

        # weights == 2 when iterating frequency is positive
        if numpy.float(slab.coords(2)) > 0.:
            assert weights > 1
            assert numpy.all(nonsig == True)
        else:
            assert weights == 1.0
            assert numpy.all(nonsig == False)
