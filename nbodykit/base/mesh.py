from nbodykit.extern.six import add_metaclass
import abc
import numpy
import logging
from pmesh.pm import ParticleMesh, RealField, ComplexField

@add_metaclass(abc.ABCMeta)
class MeshSource(object):
    """
    Base class for a source in the form of an input grid

    Subclasses must define the :func:`paint` function, which
    is abstract in this class
    """
    logger = logging.getLogger('MeshSource')

    # called by the subclasses
    def __init__(self, comm, Nmesh, BoxSize, dtype):

        # ensure self.comm is set, though usually already set by the child.
        self.comm = comm

        self.dtype = dtype
        Nmesh = numpy.array(Nmesh)
        if Nmesh.ndim == 0:
            ndim = 3
        else:
            ndim = len(Nmesh)
        _Nmesh = numpy.empty(ndim, dtype='i8')

        _Nmesh[:] = Nmesh
        self.pm = ParticleMesh(BoxSize=BoxSize,
                                Nmesh=_Nmesh,
                                dtype=self.dtype, comm=self.comm)

        self.attrs['BoxSize'] = self.pm.BoxSize.copy()
        self.attrs['Nmesh'] = self.pm.Nmesh.copy()

        self._actions = []
        self.base = None

    def view(self):
        view = object.__new__(MeshSource)
        view.comm = self.comm
        view.dtype = self.dtype
        view.pm = self.pm
        view.attrs.update(self.attrs)
        view._actions = []
        view.actions.extend(self.actions)
        view.base = self
        return view

    @property
    def actions(self):
        return self._actions

    def apply(self, func, kind='wavenumber', mode='complex'):
        view = self.view()
        view.actions.append((mode, func, kind))
        return view

    def __len__(self):
        """
        Set the length of a grid source to be 0
        """
        return 0

    def to_real_field(self):
        if self.base is not None: return self.base.to_real_field()
        return NotImplemented

    def to_complex_field(self):
        if self.base is not None: return self.base.to_complex_field()
        return NotImplemented

    def to_field(self, mode='real'):
        if mode == 'real':
            real = self.to_real_field()
            if real is NotImplemented:
                complex = self.to_complex_field()
                assert complex is not NotImplemented
                real = complex.c2r(out=Ellipsis)
                if hasattr(complex, 'attrs'):
                    real.attrs = complex.attrs
            var = real
        elif mode == 'complex':
            complex = self.to_complex_field()
            if complex is NotImplemented:
                real = self.to_real_field()
                assert real is not NotImplemented
                complex = real.r2c(out=Ellipsis)
                if hasattr(real, 'attrs'):
                    complex.attrs = real.attrs
            var = complex
        else:
            raise ValueError("mode is either real or complex, %s given" % mode)

        return var

    @property
    def attrs(self):
        """
        Dictionary storing relevant meta-data
        """
        try:
            return self._attrs
        except AttributeError:
            self._attrs = {}
            return self._attrs

    def paint(self, mode="real"):
        """
        Parameters
        ----------
        mode : string
        real or complex
        """
        if not mode in ['real', 'complex']:
            raise ValueError('mode must be "real" or "complex"')

        # add a dummy action to ensure the right mode of return value
        actions = self.actions + [(mode, )]

        # if we expect complex, be smart and use complex directly.
        var = self.to_field(mode=actions[0][0])

        if not hasattr(var, 'attrs'):
            attrs = {}
        else:
            attrs = var.attrs

        for action in actions:
            # ensure var is the right mode

            if action[0] == 'complex':
                if not isinstance(var, ComplexField):
                    var = var.r2c(out=Ellipsis)
            if action[0] == 'real':
                if not isinstance(var, RealField):
                    var = var.c2r(out=Ellipsis)

            if len(action) > 1:
                # there is a filter function
                kwargs = {}
                kwargs['func'] = action[1]
                if action[2] is not None:
                    kwargs['kind'] = action[2]
                kwargs['out'] = Ellipsis
                var.apply(**kwargs)

        var.attrs = attrs
        var.attrs.update(self.attrs)

        return var

    def preview(self, axes=None, Nmesh=None, root=0):
        """ gathers the mesh into as a numpy array, with
            (reduced resolution). The result is broadcast to
            all ranks, so this uses Nmesh ** 3 per rank.

            Parameters
            ----------
            Nmesh : int, array_like
                The desired Nmesh of the result. Be aware this function
                allocates memory to hold A full Nmesh on each rank.

            axes : int, array_like
                The axes to project the preview onto., e.g. (0, 1)

            Returns
            -------
            out : array_like
                An numpy array for the real density field.

        """
        field = self.to_field(mode='real')
        if Nmesh is None:
            Nmesh = self.pm.Nmesh

        return field.preview(Nmesh, axes=axes)

    def save(self, output, dataset='Field', mode='real'):
        """
            Save the mesh as a BigFileMesh on disk

            Parameters
            ----------
            output : str
                name of the bigfile file
            dataset : str
                name of the bigfile data set.
            mode : str
                real or complex; the form of the field to store.

        """
        import bigfile
        import warnings
        field = self.paint(mode=mode)

        with bigfile.BigFileMPI(self.pm.comm, output, create=True) as ff:
            data = numpy.empty(shape=field.size, dtype=field.dtype)
            field.ravel(out=data)
            with ff.create_from_array(dataset, data) as bb:
                if isinstance(field, RealField):
                    bb.attrs['ndarray.shape'] = field.pm.Nmesh
                    bb.attrs['BoxSize'] = field.pm.BoxSize
                    bb.attrs['Nmesh'] = field.pm.Nmesh
                elif isinstance(field, ComplexField):
                    bb.attrs['ndarray.shape'] = field.Nmesh, field.Nmesh, field.Nmesh // 2 + 1
                    bb.attrs['BoxSize'] = field.pm.BoxSize
                    bb.attrs['Nmesh'] = field.pm.Nmesh

                for key in field.attrs:
                    # do not override the above values -- they are vectors (from pm)
                    if key in bb.attrs: continue
                    value = numpy.array(field.attrs[key])
                    try:
                        bb.attrs[key] = value
                    except Exception:
                        warnings.warn("attribute %s of type %s is unsupported and lost" % (key, str(value.dtype)))
