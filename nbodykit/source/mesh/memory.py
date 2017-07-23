from nbodykit.base.mesh import MeshSource
from nbodykit import CurrentMPIComm, mockmaker
from nbodykit.utils import attrs_to_dict
from pmesh.pm import RealField, ComplexField

class MemoryMesh(MeshSource):
    """
    A MeshSource initialized from an in-memory field object, either a
    :class:`pmesh.pm.RealField` or :class:`pmesh.pm.ComplexField`.

    .. note::
        The original field object is never modified by this source.

    Parameters
    ----------
    field : :class:`~pmesh.pm.RealField`, :class:`~pmesh.pm.ComplexField`
        the :mod:`pmesh` Field object, either of real or complex type
    Nmesh : int, 3-vector of int, optional
        the desired number of cells per size on the mesh. If this is different
        than the ``Nmesh`` of the input Field, the Field will be re-sampled
    """
    def __repr__(self):
        return "MemoryMesh()"

    def __init__(self, field, Nmesh=None):

        if Nmesh is None:
            Nmesh = field.Nmesh

        MeshSource.__init__(self, field.pm.comm, Nmesh, field.BoxSize, field.pm.dtype)

        # resample the Field
        if any(field.Nmesh != self.pm.Nmesh):
            cnew = ComplexField(self.pm)
            field = field.resample(out=cnew)

            if self.comm.rank == 0: self.logger.info('resampling done')

        self.field = field

    def to_complex_field(self):
        """
        Return a copy of the (possibly re-sampled) input ComplexField
        """
        if isinstance(self.field, ComplexField):
            return self.field.copy()
        else:
            return NotImplemented

    def to_real_field(self):
        """
        Return a copy of the (possibly re-sampled) input RealField
        """
        if isinstance(self.field, RealField):
            return self.field.copy()
        else:
            return NotImplemented
