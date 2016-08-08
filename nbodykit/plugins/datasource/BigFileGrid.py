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

    def __init__(self, path, dataset):
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
        s.description = "read gridded field data from a binary file"

        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("dataset", type=str, help="the file path to load the data from")

    def read(self, pm):
        import bigfile

        f = bigfile.BigFileMPI(self.comm, self.path)
        with f[self.dataset] as ds:
            resampler.read(pm, ds, self.Nmesh, self.isfourier)

