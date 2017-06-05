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

    def __init__(self, field, Nmesh=None):
        if Nmesh is None:
            Nmesh = field.Nmesh

        MeshSource.__init__(self, field.pm.comm, Nmesh, field.BoxSize, field.pm.dtype)

        if any(field.Nmesh != self.pm.Nmesh):
            cnew = ComplexField(self.pm)
            field = field.resample(out=cnew)

            if self.comm.rank == 0: self.logger.info('resampling done')

        self.field = field

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

