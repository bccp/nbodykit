from nbodykit.core import Painter, DataSource, GridSource
from pmesh.pm import RealField, ComplexField, ParticleMesh
import numpy

class DefaultPainter(Painter):
    plugin_name = "DefaultPainter"

    def __init__(self, weight=None, frho=None, fk=None, normalize=False, setMean=None, paintbrush='cic', interlaced=False):

        self.weight     = weight
        self.frho       = frho
        self.fk         = fk
        self.normalize  = normalize
        self.setMean    = setMean
        self.paintbrush = paintbrush
        self.interlaced = interlaced

        # initialize the baseclass with the paintbrush
        super(DefaultPainter, self).__init__(paintbrush)

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "grid the density field of an input DataSource of objects, optionally "
        s.description += "using a weight for each object. "

        s.add_argument("weight", help="the column giving the weight for each object")

        s.add_argument("frho", type=str, help="A python expresion for transforming the real space density field. variables: rho. example: 1 + (rho - 1)**2")
        s.add_argument("fk", type=str, help="A python expresion for transforming the fourier space density field. variables: k, kx, ky, kz. example: exp(-(k * 0.5)**2). applied before frho ")
        s.add_argument("normalize", type=bool, help="Normalize the field to set mean == 1. Applied before fk.")
        s.add_argument("setMean", type=float, help="Set the mean. Applied after normalize.")
        s.add_argument("paintbrush", type=str, help="select a paint brush. Default is to defer to the choice of the algorithm that uses the painter.")
        s.add_argument("interlaced", type=bool, help="interlaced.")

    def paint(self, pm, datasource):
        """
        Paint the ``DataSource`` specified by ``input`` onto the 
        ``ParticleMesh`` specified by ``pm``

        Parameters
        ----------
        pm : ``ParticleMesh``
            particle mesh object that does the painting
        datasource : ``DataSource``
            the data source object representing the field to paint onto the mesh

        Returns
        -------
        stats : dict
            dictionary of statistics, usually only containing `Ntot`
        """
        stats = {}
        real = RealField(pm)

        real[:] = 0

        if isinstance(datasource, DataSource):
            # open the datasource stream (with no defaults)
            if self.interlaced:
                real2 = RealField(pm)
                real2[...] = 0

            with datasource.open() as stream:

                Nlocal = 0
                if self.weight is None:
                    for [position] in stream.read(['Position']):
                        if not self.interlaced:
                            self.basepaint(real, position, paintbrush=self.paintbrush)
                        else:
                            self.shiftedpaint(real, real2, position, paintbrush=self.paintbrush)
                        Nlocal += len(position)
                else:
                    for position, weight in stream.read(['Position', self.weight]):
                        if not self.interlaced:
                            self.basepaint(real, position, weight=weight, paintbrush=self.paintbrush)
                        else:
                            self.shiftedpaint(real, real2, position, weight=weight, paintbrush=self.paintbrush)
                        Nlocal += len(position)
            
            if self.interlaced:
                c1 = real.r2c()
                c2 = real2.r2c()

                H = pm.BoxSize / pm.Nmesh
                for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                    kH = sum(k[i] * H[i] for i in range(3))
                    s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

                c1.c2r(real)

            stats['Ntot'] = self.comm.allreduce(Nlocal)
        elif isinstance(datasource, GridSource):
            datasource.read(real)
            stats['Ntot'] = datasource.Ntot

        # apply the filters.

        mean = real.cmean()

        if self.comm.rank == 0:
            self.logger.info("Mean = %g" % mean)

        if self.normalize:
            real[...] *= 1. / mean
            mean = real.cmean()
            if self.comm.rank == 0:
                self.logger.info("Renormalized mean = %g" % mean)

        if self.setMean is not None:
            real[...] += (self.setMean - mean)

        if self.fk:
            if self.comm.rank == 0:
                self.logger.info("applying transformation fk %s" % self.fk)

            def function(k, kx, ky, kz):
                from numpy import exp, sin, cos
                def tophat(x):
                    rt = 3 * (sin(x) / x - cos(x)) / x ** 2
                    rt[x == 0] = 1
                    return rt

                return eval(self.fk)
            complex = real.r2c()
            for kk, slab in zip(complex.slabs.x, complex.slabs):
                k = sum([k ** 2 for k in kk]) ** 0.5
                slab[...] *= function(k, kk[0], kk[1], kk[2])
            complex.c2r(real)
            mean = real.cmean()
            if self.comm.rank == 0:
                self.logger.info("after fk, mean = %g" % mean)
        if self.frho:
            if self.comm.rank == 0:
                self.logger.info("applying transformation frho %s" % self.frho)

            def function(rho):
                return eval(self.frho)
            if self.comm.rank == 0:
                self.logger.info("example value before frho %g" % real.flat[0])
            for slab in real.slabs:
                slab[...] = function(slab)
            if self.comm.rank == 0:
                self.logger.info("example value after frho %g" % real.flat[0])
            mean = real.cmean()
            if self.comm.rank == 0:
                self.logger.info("after frho, mean = %g" % mean)

        return real, stats
