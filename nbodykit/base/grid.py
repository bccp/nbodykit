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
                real = complex.r2c(complex)
            var = real
        elif mode == 'complex':
            complex = self.to_complex_field()
            if complex is NotImplemented:
                real = self.to_real_field()
                assert real is not NotImplemented
                complex = real.r2c(real)
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

        for action in actions:
            # ensure var is the right mode

            if action[0] == 'complex':
                if not isinstance(var, ComplexField):
                    var = var.r2c(var)
            if action[0] == 'real':
                if not isinstance(var, RealField):
                    var = var.c2r(var)

            if len(action) > 1:
                # there is a filter function
                kwargs = {}
                kwargs['func'] = action[1]
                if action[2] is not None:
                    kwargs['kind'] = action[2]
                var.apply(**kwargs)

        if not hasattr(var, 'attrs'):
            var.attrs = {}

        var.attrs.update(self.attrs)

        # FIXME: this shall probably got to pmesh
        var.save = lambda *args, **kwargs : save(var, *args, **kwargs)

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
