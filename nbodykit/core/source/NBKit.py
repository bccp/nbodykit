from nbodykit.core import Source
from nbodykit.io.stack import FileStack
import numpy
from pmesh import window
from pmesh.pm import RealField, ComplexField

class Painter(object):
    @classmethod
    def from_config(self, d):
        return Painter(**d)

    def __init__(self, frho=None, fk=None, normalize=False, setMean=None, paintbrush='cic', interlaced=False):
        self.frho = frho
        self.fk = fk
        self.normalize = normalize
        self.setMean = setMean
        self.paintbrush = paintbrush
        self.interlaced = interlaced

    def paint(self, stream, pm):
        paintbrush = window.methods[self.paintbrush]

        real = RealField(pm)
        real[:] = 0

        if self.interlaced:
            real2 = RealField(pm)
            real2[...] = 0

        Nlocal = 0
        for chunk in stream.read(['Position', 'Weight', 'Selection']):

            [position, weight, selection] = chunk

            if weight is None:
                weight = numpy.ones(len(position))

            if selection is not None:
                position = position[selection]
                weight = weight[selection]

            Nlocal += len(position)

            if not self.interlaced:
                lay = pm.decompose(position, smoothing=0.5 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)
                real.paint(position, mass=weight, method=paintbrush)
            else:
                lay = pm.decompose(position, smoothing=1.0 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)

                shifted = pm.affine.shift(shift)

                real.paint(position, mass=weight, method=paintbrush)
                real2.paint(position, mass=weight, method=paintbrush, transform=shifted)
                c1 = real.r2c()
                c2 = real2.r2c()

                H = pm.BoxSize / pm.Nmesh
                for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                    kH = sum(k[i] * H[i] for i in range(3))
                    s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

                c1.c2r(real)

        real.shotnoise = numpy.prod(pm.BoxSize) / pm.comm.allreduce(Nlocal)
        return real

    def transform(self, stream, real):
        comm = real.pm.comm
        logger = stream.logger

        mean = comm.allreduce(real.sum(dtype='f8')) / real.Nmesh.prod()

        if comm.rank == 0:
            logger.info("Mean = %g" % mean)

        if self.normalize:
            real[...] *= 1. / mean
            mean = comm.allreduce(real.sum(dtype='f8')) / real.Nmesh.prod()
            if comm.rank == 0:
                logger.info("Renormalized mean = %g" % mean)

        if self.setMean is not None:
            real[...] += (self.setMean - mean)

        if self.fk:
            if comm.rank == 0:
                logger.info("applying transformation fk %s" % self.fk)

            def function(k, kx, ky, kz):
                from numpy import exp, sin, cos
                return eval(self.fk)
            complex = real.r2c()
            for kk, slab in zip(complex.slabs.x, complex.slabs):
                k = sum([k ** 2 for k in kk]) ** 0.5
                slab[...] *= function(k, kk[0], kk[1], kk[2])
            complex.c2r(real)
            mean = comm.allreduce(real.sum(dtype='f8')) / real.Nmesh.prod()
            if comm.rank == 0:
                logger.info("after fk, mean = %g" % mean)
        if self.frho:
            if comm.rank == 0:
                logger.info("applying transformation frho %s" % self.frho)

            def function(rho):
                return eval(self.frho)
            if comm.rank == 0:
                logger.info("example value before frho %g" % real.flat[0])
            for slab in real.slabs:
                slab[...] = function(slab)
            if comm.rank == 0:
                logger.info("example value after frho %g" % real.flat[0])
            mean = comm.allreduce(real.sum(dtype='f8')) / real.Nmesh.prod()
            if comm.rank == 0:
                logger.info("after frho, mean = %g" % mean)

class NBKitSource(Source):
    plugin_name = "Source.NBKit"

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

