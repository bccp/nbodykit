from nbodykit.base.mesh import MeshSource
from nbodykit import CurrentMPIComm, mockmaker
from nbodykit.utils import attrs_to_dict
from pmesh.pm import RealField, ComplexField

class FieldMesh(MeshSource):
    """
    A MeshSource initialized from an in-memory Field object, either a
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
        return "FieldMesh()"

    def __init__(self, field):

        MeshSource.__init__(self, field.pm.comm, field.Nmesh, field.BoxSize, field.pm.dtype)
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
