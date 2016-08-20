from nbodykit.core import Source
from nbodykit.core.source import Painter

from nbodykit.io.stack import FileStack
import numpy
from pmesh import window

class ParticleSource(Source):
    plugin_name = "Source.Particle"

    def __init__(self, filetype, path, args={}, transform={}, attrs={}, painter=Painter(),
        enable_dask=False):

        # cannot do this in the module because the module file is ran before plugin_manager
        # is init.
        from nbodykit import plugin_manager
        filetype = plugin_manager.get_plugin(filetype)

        self.logger.info("Extra arguments to FileType: %s " % args)

        self.cat = FileStack(path, filetype, **args)

        self._attrs = {}
        self._attrs.update(self.cat.attrs)
        self._attrs.update(attrs)

        for key in self.attrs.keys():
            self.attrs[key] = numpy.asarray(self.attrs[key])

        self.transform = transform
        self.enable_dask = enable_dask
        self.painter = painter
        self.logger.info("attrs = %s" % self.attrs)

        if enable_dask:
            import dask

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

        s.add_argument("enable_dask", type=bool, help="use dask")

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
        if self.enable_dask:
            import dask.array as da
            ds = da.from_array(self.cat, 1024 * 32)

            def ev(column):
                if column in self.transform:
                    g = {'ds' : ds, 'attrs' : self.attrs}
                    return eval(self.transform[column], g)[start:end]
                elif column in ds:
                    return ds[column][start:end]
                else:
                    return None

            yield da.compute(*[ev(key) for key in columns])
        else:
            ds = self.cat[start:end]
            def ev(column):
                if column in self.transform:
                    g = {'ds' : ds, 'attrs' : self.attrs}
                    return eval(self.transform[column], g)
                elif column in ds.dtype.names:
                    return ds[column]
                else:
                    return None

            yield [ev(key) for key in columns]

    def paint(self, pm):
        if self.painter is None:
            raise ValueError("No painter is provided")
        real = self.painter.paint(self, pm)

        # apply transformations
        self.painter.transform(self, real)
        return real

