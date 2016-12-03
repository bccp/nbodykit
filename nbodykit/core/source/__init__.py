from ...plugins import PluginBase, PluginBaseMeta, MetaclassWithHooks
from ...plugins.hooks import attach_cosmo
from ...extern.six import add_metaclass

from abc import abstractmethod, abstractproperty
import numpy

# attach the cosmology to data sources
SourceMeta = MetaclassWithHooks(PluginBaseMeta, attach_cosmo)

@add_metaclass(SourceMeta)
class Source(PluginBase):
    """
    A base class to represent an object that combines the processes
    of reading / generating data and painting to a RealField
    """

    @staticmethod
    def compute(*args, **kwargs):
        """
        Our version of :func:`dask.compute` that computes
        multiple delayed dask collections at once
        
        Parameters
        -----------
        args : object
            Any number of objects. If the object is a dask 
            collection, it's computed and the result is returned. 
            Otherwise it's passed through unchanged.
        
        Notes
        -----
        The dask default optimizer induces too many (unnecesarry) 
        IO calls -- we turn this off feature off by default.
        
        Eventually we want our own optimizer probably.
        """
        import dask
        
        # XXX find a better place for this function
        kwargs.setdefault('optimize_graph', False)
        return dask.compute(*args, **kwargs)

    @property
    def BoxSize(self):
        """
        A 3-vector specifying the size of the box for this source
        """
        BoxSize = numpy.array([1, 1, 1.], dtype='f8')
        BoxSize[:] = self.attrs['BoxSize']
        return BoxSize

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
from nbodykit.plugins.schema import attribute

class Painter(object):
    """
    Painter object to help Sources convert results from Source.read to a RealField 
    """
    @attribute('interlaced', type=bool, help="whether to use interlacing to minimize aliasing effects")
    @attribute('paintbrush', type=str, choices=list(window.methods), help="The paint brush to use when interpolating the field to the mesh")
    @attribute('setMean', type=float, help="Set the mean of the real-space field. Applied after normalize.")
    @attribute('normalize', type=bool, help="Normalize the field to set mean == 1. Applied before fk.")
    @attribute('fk', type=str, help="A python expresion for transforming the fourier space density field. variables: k, kx, ky, kz. example: exp(-(k * 0.5)**2). applied before frho")
    @attribute('frho', type=str, help="A python expresion for transforming the real space density field. variables: rho. example: 1 + (rho - 1)**2")
    def __init__(self, frho=None, fk=None, normalize=False, setMean=None, paintbrush='cic', interlaced=False):
        
        self.frho       = frho
        self.fk         = fk
        self.normalize  = normalize
        self.setMean    = setMean
        self.paintbrush = paintbrush
        self.interlaced = interlaced

    @classmethod
    def from_config(cls, d):
        """
        Initialize the class from a dictionary of parameters
        """
        return cls(**d)
        
    def paint(self, source, pm):
        """
        Paint the input `source` to the mesh specified by `pm`
        
        Parameters
        ----------
        source : `Source` or a subclass
            the source object from which the default 
        pm : pmesh.pm.ParticleMesh
            the particle mesh object
        
        Returns
        -------
        real : pmesh.pm.RealField
            the painted real field
        """
        Nlocal = 0 # number of particles read on local rank
        
        # the paint brush window
        paintbrush = window.methods[self.paintbrush]

        # initialize the RealField to returns
        real = RealField(pm)
        real[:] = 0

        # need 2nd field if interlacing
        if self.interlaced:
            real2 = RealField(pm)
            real2[...] = 0

        # read the necessary data (as dask arrays)
        columns = ['Position', 'Weight', 'Selection']
        if not all(col in source for col in columns):
            raise ValuError("source does not contain columns: %s" %str(columns))
        Position, Weight, Selection = source.read(columns)

        # size of the data the local rank is responsible for
        N = len(Position)
        
        # paint data in chunks on each rank
        chunksize = 1024 ** 2
        for i in range(0, N, chunksize):

            s = slice(i, i + chunksize)
            position, weight, selection = source.compute(Position[s], Weight[s], Selection[s])

            if weight is None:
                weight = numpy.ones(len(position))

            if selection is not None:
                position = position[selection]
                weight   = weight[selection]

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

    def transform(self, source, real):
        """
        Apply (in-place) transformations to the real-space field 
        specified by `real`
        """
        comm = real.pm.comm
        logger = source.logger

        # mean of the field
        mean = comm.allreduce(real.sum(dtype='f8')) / real.Nmesh.prod()
        if comm.rank == 0: logger.info("Mean = %g" % mean)

        # normalize the field by dividing out the mean
        if self.normalize:
            real[...] *= 1. / mean
            mean = comm.allreduce(real.sum(dtype='f8')) / real.Nmesh.prod()
            if comm.rank == 0: logger.info("Renormalized mean = %g" % mean)

        # explicity set the mean
        if self.setMean is not None:
            real[...] += (self.setMean - mean)

        # apply transformation in Fourier space
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
        
        # apply transformation in real-space
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

