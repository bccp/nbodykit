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


    def paint(self, kind="real"):
        """
        Parameters
        ----------
        kind : string
        real or complex
        """
        real = self.source.paint(self.pm)
        if kind == 'real':
            return real
        elif kind == 'complex':
            return real.r2c()
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

