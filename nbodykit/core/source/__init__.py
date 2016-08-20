from ...plugins import PluginBase, PluginBaseMeta, MetaclassWithHooks
from ...plugins.hooks import attach_cosmo
from ...extern.six import add_metaclass

from abc import abstractmethod, abstractproperty
import numpy

# attach the cosmology to data sources
SourceMeta = MetaclassWithHooks(PluginBaseMeta, attach_cosmo)

@add_metaclass(SourceMeta)
class Source(PluginBase):

    @abstractproperty
    def columns(self):
        return []

    @abstractproperty
    def attrs(self):
        return {}

    @abstractmethod
    def read(self, columns):
        yield []

    @abstractmethod
    def paint(self, pm):
        pass

from pmesh import window
from pmesh.pm import RealField, ComplexField

class Painter(object):
    """ Painter object helps to Sources to convert results from Source.read to a RealField """
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

