from abc import abstractmethod, abstractproperty
import numpy
import logging

from pmesh.pm import RealField, ComplexField, ParticleMesh

class Field(object):
    logger = logging.getLogger('Field')

    def __init__(self, source, Nmesh, dtype='f4', comm=None):

        self.Nmesh = Nmesh
        self.source = source
        self.dtype = dtype
        self.comm = source.comm

        self.pm = ParticleMesh(BoxSize=self.source.BoxSize,
                                Nmesh=[self.Nmesh]*3,
                                dtype=self.dtype, comm=self.comm)

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

    def paint(self, kind="real"):
        """
        Parameters
        ----------
        kind : string
        real or complex
        """
        real = self.source.paint(self.pm)
        var = real
        for action in self.actions:
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
            if isinstance(var, RealField):
                return var
            else:
                return var.c2r()
        elif kind == 'complex':
            if isinstance(var, RealField):
                return var.r2c()
            else:
                return var
        else:
            raise ValueError('kind must be "real" or "complex"')

    def save(self, field, output, dataset="Field"):
        import bigfile
        with bigfile.BigFileMPI(self.comm, output, create=True) as ff:
            data = numpy.empty(shape=field.size, dtype=field.dtype)
            field.sort(out=data)
            if isinstance(field, RealField):
                with ff.create_from_array(dataset, data) as bb:
                    bb.attrs['ndarray.shape'] = self.pm.Nmesh
                    bb.attrs['BoxSize'] = self.pm.BoxSize
                    bb.attrs['Nmesh'] = self.pm.Nmesh
            elif isinstance(field, ComplexField):
                with ff.create_from_array(dataset, data) as bb:
                    bb.attrs['ndarray.shape'] = self.Nmesh, self.Nmesh, self.Nmesh // 2 + 1
                    bb.attrs['BoxSize'] = self.pm.BoxSize
                    bb.attrs['Nmesh'] = self.pm.Nmesh

