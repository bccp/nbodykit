from nbodykit.extensionpoints import GridSource
from nbodykit import resampler
import numpy

class BigFileGridSource(GridSource):
    """
    Class to read field gridded data from a binary file

    Notes
    -----
    * Reading is designed to be done by `GridPainter`, which
      reads gridded quantity straight into the `ParticleMesh`
    """
    plugin_name = "BigFileGrid"

    def __init__(self, path, dataset, frho=None, normalize=False, fk=None):
        import bigfile
        f = bigfile.BigFileMPI(self.comm, self.path)
        self.dataset = dataset
        self.path = path
        with f[self.dataset] as d:
            self.BoxSize = d.attrs['BoxSize']
            self.Nmesh = int(d.attrs['Nmesh'][0])
            if 'Ntot' in d.attrs:
                self.Ntot = d.attrs['Ntot'][0]
            else:
                self.Ntot = 0

            # Is this a complex field or a real field?
            if d.dtype.kind == 'c':
                self.isfourier = True
            else:
                self.isfourier = False

    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "read gridded field data from a binary file. Fourier resampling is applied if necessary."

        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("dataset", type=str, help="the file path to load the data from")
        s.add_argument("frho", type=str, help="A python expresion for transforming the real space density field. variables: rho. example: 1 + (rho - 1)**2")
        s.add_argument("fk", type=str, help="A python expresion for transforming the fourier space density field. variables: k. example: exp(-(k * 0.5)**2) ")
        s.add_argument("normalize", type=bool, help="Normalize the field to set mean == 1")

    def read(self, pm):
        import bigfile
        if self.comm.rank == 0:
            self.logger.info("Reading from Nmesh = %d to Nmesh = %d" %(self.Nmesh, pm.Nmesh))

        f = bigfile.BigFileMPI(self.comm, self.path)
        with f[self.dataset] as ds:
            resampler.read(pm, ds, self.Nmesh, self.isfourier)
        mean = self.comm.allreduce(pm.real.sum(dtype='f8')) / pm.Nmesh ** 3.

        if self.comm.rank == 0:
            self.logger.info("Mean = %g" % mean)

        if self.normalize:
            pm.real *= 1. / mean
            mean = self.comm.allreduce(pm.real.sum(dtype='f8')) / pm.Nmesh ** 3.
            if self.comm.rank == 0:
                self.logger.info("Renormalized mean = %g" % mean)

        if self.fk:
            if self.comm.rank == 0:
                self.logger.info("applying transformation fk %s" % self.fk)

            def function(rho):
                return eval(self.frho)
            pm.r2c()
            k = (pm.k[0] ** 2 + pm.k[1] ** 2 + pm.k[2] ** 2) ** 0.5
            pm.complex[...] *= function(k)
            pm.c2r()

        if self.frho:
            if self.comm.rank == 0:
                self.logger.info("applying transformation frho %s" % self.frho)

            def function(rho):
                return eval(self.frho)
            if self.comm.rank == 0:
                self.logger.info("example value before frho %g" % pm.real.flat[0])
            pm.real[...] = function(pm.real)
            if self.comm.rank == 0:
                self.logger.info("example value after frho %g" % pm.real.flat[0])

