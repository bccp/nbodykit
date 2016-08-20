from nbodykit.core import Source
from nbodykit.core.source import Painter

from bigfile import BigFileMPI
from pmesh.pm import RealField, ComplexField, ParticleMesh
import numpy
from pmesh import window

class GridSource(Source):
    plugin_name = "Source.Grid"

    def __init__(self, path, dataset, attrs={}, painter=Painter()):

        # cannot do this in the module because the module file is ran before plugin_manager
        # is init.

        self.cat = BigFileMPI(comm=self.comm, filename=path)[dataset]

        self._attrs = {}
        self._attrs.update(self.cat.attrs)
        self._attrs.update(attrs)

        for key in self.attrs.keys():
            self.attrs[key] = numpy.asarray(self.attrs[key])

        if self.comm.rank == 0:
            self.logger.info("attrs = %s" % self.attrs)

        self.painter= painter
        self.Nmesh = self.attrs['Nmesh'].squeeze()

        if 'shotnoise' in self.attrs:
            self.shotnoise = self.attrs['shotnoise'].squeeze()
        else:
            self.shotnoise = 0

        if self.cat.dtype.kind == 'c':
            self.isfourier = True
        else:
            self.isfourier = False

    @property
    def columns(self):
        return []

    @property
    def attrs(self):
        return self._attrs

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "read snapshot files a multitype file"
        s.add_argument("path", help="the file path to load the data from")
        s.add_argument("dataset", help="dataset")
        s.add_argument("attrs", type=dict, help="override attributes from the file")

        s.add_argument("painter", type=Painter.from_config, help="painter parameters")

        # XXX for painting needs some refactoring
        s.add_argument("painter.paintbrush", choices=list(window.methods.keys()), help="paintbrush")
        s.add_argument("painter.frho", type=str, help="A python expresion for transforming the real space density field. variables: rho. example: 1 + (rho - 1)**2")
        s.add_argument("painter.fk", type=str, help="A python expresion for transforming the fourier space density field. variables: k, kx, ky, kz. example: exp(-(k * 0.5)**2). applied before frho ")
        s.add_argument("painter.normalize", type=bool, help="Normalize the field to set mean == 1. Applied before fk.")
        s.add_argument("painter.setMean", type=float, help="Set the mean. Applied after normalize.")
        s.add_argument("painter.interlaced", type=bool, help="interlaced.")

    def read(self, columns):
        yield [None for key in columns]

    def paint(self, pm):

        if self.painter is None:
            raise ValueError("No painter is provided")
        real = RealField(pm)

        if any(pm.Nmesh != self.Nmesh):
            pmread = ParticleMesh(BoxSize=pm.BoxSize, Nmesh=(self.Nmesh, self.Nmesh, self.Nmesh),
                    dtype='f4', comm=self.comm)
        else:
            pmread = real.pm

        ds = self.cat

        if self.isfourier:
            if self.comm.rank == 0:
                self.logger.info("reading complex field")
            complex2 = ComplexField(pmread)
            assert self.comm.allreduce(complex2.size) == ds.size
            start = sum(self.comm.allgather(complex2.size)[:self.comm.rank])
            end = start + complex2.size
            complex2.unsort(ds[start:end])
            complex2.resample(real)
        else:
            if self.comm.rank == 0:
                self.logger.info("reading real field")
            real2 = RealField(pmread)
            start = sum(self.comm.allgather(real2.size)[:self.comm.rank])
            end = start + real2.size
            real2.unsort(ds[start:end])
            real2.resample(real)

        real.shotnoise = self.shotnoise

        # apply transformations
        self.painter.transform(self, real)
        return real

