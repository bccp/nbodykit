from nbodykit.core import GridSource
import numpy
from pmesh.pm import ParticleMesh, RealField, ComplexField

class BigFileGridSource(GridSource):
    """
    Class to read field gridded data from a binary file

    Notes
    -----
    * Reading is designed to be done by `GridPainter`, which
      reads gridded quantity straight into the `ParticleMesh`
    """
    plugin_name = "BigFileGrid"

    def __init__(self, path, dataset):
        
        self.path = path
        self.dataset = dataset
        
        import bigfile
        f = bigfile.BigFileMPI(self.comm, self.path)

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
    def fill_schema(cls):
        s = cls.schema
        s.description = "read gridded field data from a binary file. Fourier resampling is applied if necessary."

        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("dataset", type=str, help="the file path to load the data from")

    def read(self, real):
        import bigfile
        if self.comm.rank == 0:
            self.logger.info("Reading from Nmesh = %d to Nmesh = %d" %(self.Nmesh, real.Nmesh[0]))

        if any(real.Nmesh != self.Nmesh):
            pmread = ParticleMesh(BoxSize=real.BoxSize, Nmesh=(self.Nmesh, self.Nmesh, self.Nmesh),
                    dtype='f4', comm=self.comm)
        else:
            pmread = real.pm

        f = bigfile.BigFileMPI(self.comm, self.path)

        with f[self.dataset] as ds:
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
