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

    @property
    def actions(self):
        return self._actions

    def apply(self, func, kind='wavenumber', mode='complex'):
        self.actions.append((mode, func, kind))

    def __len__(self):
        """
        Set the length of a grid source to be 0
        """
        return 0

    def to_real_field(self):
        return NotImplemented

    def to_complex_field(self):
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

        # FIXME: this shall probably got to pmesh
        var.save = lambda *args, **kwargs : save(var, *args, **kwargs)
        return var

    def preview(self, Nmesh=None, root=0):
        """ gathers the mesh into as a numpy array, with
            (reduced resolution). The result is broadcast to
            all ranks, so this uses Nmesh ** 3 per rank.

            Parameters
            ----------
            Nmesh : int, array_like
                The desired Nmesh of the result. Be aware this function
                allocates memory to hold A full Nmesh on each rank.

            Returns
            -------
            out : array_like
                An numpy array for the real density field.

        """
        field = self.to_field(mode='real')
        if any(Nmesh != self.attrs['Nmesh']):
            _Nmesh = self.attrs['Nmesh'].copy()
            _Nmesh[:] = Nmesh
            pm = ParticleMesh(BoxSize=self.attrs['BoxSize'],
                                Nmesh=_Nmesh,
                                dtype=self.dtype, comm=self.comm)
            out = pm.create(mode='real')
            field.resample(out)
        else:
            out = field

        from mpi4py import MPI # for inplace

        r = numpy.zeros(out.cshape, out.dtype)
        r[out.slices] = out.value
        self.comm.Allreduce(MPI.IN_PLACE, r)
        return r

def save(self, output, dataset='Field'):
    import bigfile
    import warnings
    with bigfile.BigFileMPI(self.pm.comm, output, create=True) as ff:
        data = numpy.empty(shape=self.size, dtype=self.dtype)
        self.sort(out=data)
        with ff.create_from_array(dataset, data) as bb:
            if isinstance(self, RealField):
                bb.attrs['ndarray.shape'] = self.pm.Nmesh
                bb.attrs['BoxSize'] = self.pm.BoxSize
                bb.attrs['Nmesh'] = self.pm.Nmesh
            elif isinstance(self, ComplexField):
                bb.attrs['ndarray.shape'] = self.Nmesh, self.Nmesh, self.Nmesh // 2 + 1
                bb.attrs['BoxSize'] = self.pm.BoxSize
                bb.attrs['Nmesh'] = self.pm.Nmesh

            for key in self.attrs:
                # do not override the above values -- they are vectors (from pm)
                if key in bb.attrs: continue
                value = numpy.array(self.attrs[key])
                try:
                    bb.attrs[key] = value
                except TypeError:
                    warnings.warn("attribute %s of type %s is unsupported and lost" % (key, str(value.dtype)))
