from nbodykit.base.mesh import MeshSource
from nbodykit import CurrentMPIComm, mockmaker
from nbodykit.utils import attrs_to_dict
from pmesh.pm import RealField, ComplexField

class MemoryMesh(MeshSource):
    """
    A source to adapt an in memory pmesh Field object, ``RealField`` or ``ComplexField`` as a MeshSource
    of nbodykit. 

    The original field object is never modified by this source.
    """

    def __repr__(self):
        return "MemoryMesh()"

    def __init__(self, field):
        self.field = field
        MeshSource.__init__(self, field.pm.comm, field.Nmesh, field.BoxSize, field.pm.dtype)

    def to_complex_field(self):
        if isinstance(self.field, ComplexField):
            return self.field.copy()
        else:
            return NotImplemented

    def to_real_field(self):
        if isinstance(self.field, RealField):
            return self.field.copy()
        else:
            return NotImplemented

