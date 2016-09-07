from nbodykit.core import Source
from nbodykit.core.source import Painter

from nbodykit.io.stack import FileStack
import numpy
from pmesh import window
import dask.array as da

class ParticleSource(Source):
    plugin_name = "Source.Particle"

    def __init__(self, filetype, path, args={}, transform={}, attrs={}, painter=Painter()):

        # cannot do this in the module because the module file is ran before plugin_manager
        # is init.
        from nbodykit import plugin_manager
        filetype = plugin_manager.get_plugin(filetype)

        self.cat = FileStack(path, filetype, **args)

        self._attrs = {}
        self._attrs.update(self.cat.attrs)
        self._attrs.update(attrs)

        for key in self.attrs.keys():
            self.attrs[key] = numpy.asarray(self.attrs[key])

        if self.comm.rank == 0:
            self.logger.info("Extra arguments to FileType: %s " % args)
            self.logger.info("attrs = %s" % self.attrs)

        self.transform = {
                "Selection" : "da.ones(cat.size, dtype='?', chunks=10000)",
                "Weight" : "da.ones(cat.size, dtype='f4', chunks=10000)",
        }

        self.transform.update(transform)
        self.painter = painter

        self.ds = dict([(column, self.cat.get_dask(column)) for column in self.cat.dtype.names])

    @property
    def columns(self):
        return sorted(set(self.cat.dtype.names) + set(self.transform.keys()))

    @property
    def attrs(self):
        return self._attrs

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "read snapshot files a multitype file"
        s.add_argument("filetype", help="the file path to load the data from")
        s.add_argument("path", help="the file path to load the data from")
        s.add_argument("args", type=dict, help="the file path to load the data from")
        s.add_argument("transform", type=dict, help="data transformation")
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
        # XXX: make this a iterator? 
        start = self.comm.rank * self.cat.size // self.comm.size
        end = (self.comm.rank  + 1) * self.cat.size // self.comm.size

        def ev(column):
            if column in self.transform:
                g = {'ds' : self.ds, 'attrs' : self.attrs, 'cat' : self.cat, 'da' : da}
                return eval(self.transform[column], g)[start:end]
            elif column in self.ds:
                return self.ds[column][start:end]
            else:
                raise KeyError("column `%s` is neither provided as a transformed column nor in the file" % column)

        return [ev(key) for key in columns]

    def paint(self, pm):
        if self.painter is None:
            raise ValueError("No painter is provided")
        real = self.painter.paint(self, pm)

        # apply transformations
        self.painter.transform(self, real)
        return real

