from nbodykit.extern.six import add_metaclass
import abc
import numpy
import logging

from pmesh.pm import ParticleMesh, RealField, ComplexField

@add_metaclass(abc.ABCMeta)
class GridSource(object):
    """
    Base class for a source in the form of an input grid

    Subclasses must define the :func:`paint` function, which
    is abstract in this class
    """
    logger = logging.getLogger('GridSource')

    # called by the subclasses
    def __init__(self, comm, Nmesh, BoxSize, dtype):

        # ensure self.comm is set, though usually already set by the child.
        self.comm = comm

        self.dtype = dtype
        _Nmesh = numpy.empty(3, dtype='i8')
        _Nmesh[:] = Nmesh
        self.pm = ParticleMesh(BoxSize=BoxSize,
                                Nmesh=_Nmesh,
                                dtype=self.dtype, comm=self.comm)

        self.attrs['BoxSize'] = self.pm.BoxSize.copy()
        self.attrs['Nmesh'] = self.pm.Nmesh.copy()

        if self.comm.rank == 0:
            self.logger.info("attrs = %s" % self.attrs)


        self._actions = []

    @property
    def actions(self):
        return self._actions

    def r2c(self):
        self.actions.append(('r2c',))

    def c2r(self):
        self.actions.append(('c2r',))

    def apply(self, func, kind=None):
        self.actions.append(('apply', func, kind))

    def __len__(self):
        """
        Set the length of a grid source to be 0
        """
        return 0

    def to_real_field(self):
        return NotImplemented

    def to_complex_field(self):
        return NotImplemented

    def to_field(self, kind='real'):
        if kind == 'real':
            real = self.to_real_field()
            if real is NotImplemented:
                complex = self.to_complex_field()
                assert complex is not NotImplemented
                real = complex.r2c(complex)
            var = real
        elif kind == 'complex':
            complex = self.to_complex_field()
            if complex is NotImplemented:
                real = self.to_real_field()
                assert real is not NotImplemented
                complex = real.r2c(real)
            var = complex
        else:
            raise ValueError("kind is either real or complex")

        if not hasattr(var, 'attrs'):
            var.attrs = {}

        var.attrs.update(self.attrs)

        # FIXME: this shall probably got to pmesh
        var.save = lambda *args, **kwargs : save(var, *args, **kwargs)

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

    def paint(self, kind="real"):
        """
        Parameters
        ----------
        kind : string
        real or complex
        """

        # these methods are implemented in the base classes of sources.
        # it may be wise to move them here.

        # FIXME: if there is a c2r be smart and use complex directly.
        var = self.to_field(kind='real')

        attrs = var.attrs
        save = var.save

        for action in remove_roundtrips(self.actions):

            if   action[0] == 'r2c':
                var = var.r2c()
            elif action[0] == 'c2r':
                var = var.c2r()
            elif action[0] == 'apply':
                kwargs = {}
                kwargs['func'] = action[1]
                if action[2] is not None:
                    kwargs['kind'] = action[2]
                var.apply(**kwargs)

        if kind == 'real':
            if not isinstance(var, RealField):
                var = var.c2r()
        elif kind == 'complex':
            if not isinstance(var, ComplexField):
                var = var.r2c()
        else:
            raise ValueError('kind must be "real" or "complex"')

        var.attrs = attrs
        var.save = save

        return var

def save(self, output, dataset='Field'):
    import bigfile
    with bigfile.BigFileMPI(self.pm.comm, output, create=True) as ff:
        data = numpy.empty(shape=self.size, dtype=self.dtype)
        self.sort(out=data)
        if isinstance(self, RealField):
            with ff.create_from_array(dataset, data) as bb:
                bb.attrs['ndarray.shape'] = self.pm.Nmesh
                bb.attrs['BoxSize'] = self.pm.BoxSize
                bb.attrs['Nmesh'] = self.pm.Nmesh
                for key in self.attrs:
                    # do not override the above values -- they are vectors (from pm)
                    if key in bb.attrs: continue
                    bb.attrs[key] = self.attrs[key]
        elif isinstance(self, ComplexField):
            with ff.create_from_array(dataset, data) as bb:
                bb.attrs['ndarray.shape'] = self.Nmesh, self.Nmesh, self.Nmesh // 2 + 1
                bb.attrs['BoxSize'] = self.pm.BoxSize
                bb.attrs['Nmesh'] = self.pm.Nmesh
                for key in self.attrs:
                    if key in bb.attrs: continue
                    bb.attrs[key] = self.attrs[key]
        
def remove_roundtrips(actions):
    i = 0
    while i < len(actions):
        action = actions[i]
        next = actions[i + 1] if i < len(actions) - 1 else ('',)
        if action[0] == 'r2c' and next[0] == 'c2r':
            i = i + 2
            continue
        if action[0] == 'c2r' and next[0] == 'r2c':
            i = i + 2
            continue
        yield action
        i = i + 1

