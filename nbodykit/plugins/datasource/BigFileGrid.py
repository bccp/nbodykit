from nbodykit.extensionpoints import GridSource
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
            self.Nmesh = d.attrs['Nmesh'][0]
            self.Ntot = d.attrs['Ntot'][0]

    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "read gridded field data from a binary file"

        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("dataset", type=str, help="the file path to load the data from")

    def read(self, pm):
        if pm.Nmesh != self.Nmesh:
            raise ValueError("The Grid has a mesh size of %d, but ParticleMesh has %d" % (self.Nmesh, pm.Nmesh))

        import bigfile
        import mpsort
        f = bigfile.BigFileMPI(self.comm, self.path)

        istart = pm.partition.local_i_start

        x3d = numpy.empty(pm.real.shape, dtype='f4')

        ind = numpy.zeros(x3d.shape, dtype='i8')

        for d in range(3):
            i = numpy.arange(istart[d], istart[d] + x3d.shape[d])
            i = i.reshape([-1 if dd == d else 1 for dd in range(3)])
            ind[...] *= pm.Nmesh
            ind[...] += i

        ind = ind.ravel()
        x3d = x3d.ravel()

        start = sum(self.comm.allgather(x3d.size)[:self.comm.rank])
        end = start + x3d.size

        originind = numpy.arange(start, end, dtype='i8')

        mpsort.sort(originind, orderby=ind, comm=self.comm)

        with f[self.dataset] as ds:
            x3d[:] = ds[start: end]

        mpsort.sort(x3d, orderby=originind, comm=self.comm)

        pm.real.flat[:] = x3d
