from nbodykit.base.mesh import MeshSource

from nbodykit import CurrentMPIComm
from nbodykit.utils import attrs_to_dict
from pmesh.pm import RealField, ComplexField
import numpy

class ArrayMesh(MeshSource):
    """
    A source to adapt an in-memory array, as a MeshSource of nbodykit.

    The in-memory array must be fully hosted by the root rank.
    """

    def __repr__(self):
        return "ArrayMesh()"

    @CurrentMPIComm.enable
    def __init__(self, array, BoxSize, comm=None, root=0, **kwargs):
        if comm.rank == root:
            array = numpy.array(array)
            if array.dtype.kind == 'c':
                # transform to real for the correct shape
                array = numpy.fft.irfftn(array)
                array[...] *= numpy.prod(array.shape)
            shape = array.shape
            dtype = array.dtype
        else:
            array, dtype, shape = [None] * 3

        dtype = comm.bcast(dtype, root=root)
        shape = comm.bcast(shape, root=root)

        assert len(shape) in (2, 3)

        Nmesh = shape

        empty = numpy.empty((0,), dtype)

        MeshSource.__init__(self, comm, Nmesh, BoxSize, empty.real.dtype)

        self.field = self.pm.create(mode='real')

        if comm.rank != root:
            array = empty # ignore data from other ranks.
        else:
            array = array.ravel()

        # fill the field with the array
        self.field.unravel(array)

    def to_real_field(self):
        if isinstance(self.field, RealField):
            return self.field.copy()
        else:
            return NotImplemented

