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
        if pm.Nmesh == self.Nmesh:
            self.directread(pm)
        else:
            self.resampleread(pm)

    def directread(self, pm):
        if pm.Nmesh != self.Nmesh:
            raise ValueError("The Grid has a mesh size of %d, but ParticleMesh has %d" % (self.Nmesh, pm.Nmesh))
        import bigfile
        import mpsort
        f = bigfile.BigFileMPI(self.comm, self.path)

        ind = build_index(
                [ numpy.arange(s, s + n)
                  for s, n in zip(pm.partition.local_i_start,
                                pm.real.shape)
                ], [self.Nmesh, self.Nmesh, self.Nmesh])

        with f[self.dataset] as ds:
            start, end = mpsort.globalrange(ind.flat, self.comm)
            data = ds[start:end]
            mpsort.permute(data, ind.flat, self.comm, out=pm.real.flat)

    def resampleread(self, pm):
        oldpm = ParticleMesh(self.BoxSize, self.Nmesh, dtype='f4', comm=self.comm)

        self.directread(oldpm)

        oldpm.r2c()

        if self.Nmesh >= pm.Nmesh:
            downsample(oldpm, pm)
        else:
            upsample(oldpm, pm)

        pm.c2r()

def upsample(pmsrc, pmdest):
    assert pmdest.Nmesh >= pmsrc.Nmesh
    # indtable stores the index in pmsrc for the mode in pmdest
    # since pmdest > pmsrc, some items are -1
    indtable = reindex(pmdest, pmsrc)

    ind = build_index(
            [ indtable[numpy.arange(s, s + n)]
              for s, n in zip(pm.partition.local_o_start,
                            pm.complex.shape)
            ], [Nmesh, Nmesh, Nmesh // 2 + 1])

    pmdest.complex[:] = 0

    # fill the points that has values in pmsrc
    mask = ind >= 0
    # their indices
    argind = ind[mask]
    # take the data
    data = mpsort.take(pmsrc.complex.flat, argind, pmsrc.comm)
    # fill in the value
    pmdest.complex[mask] = data

def downsample(pmsrc, pmdest):
    assert pmdest.Nmesh <= pmsrc.Nmesh
    # indtable stores the index in pmsrc for the mode in pmdest
    # since pmdest < pmsrc, all items are alright.
    indtable = reindex(pmdest, pmsrc)

    ind = build_index(
            [ indtable[numpy.arange(s, s + n)]
              for s, n in zip(pm.partition.local_o_start,
                            pm.complex.shape)
            ], [Nmesh, Nmesh, Nmesh // 2 + 1])

    mpsort.take(pmsrc.complex.flat, ind.flat, pmsrc.comm, out=pmdest.flat)

def build_index(indices, fullshape):
    """
        Build a linear index array based on indices on an array of fullshape.
        This is similar to numpy.ravel_multi_index.

        index value of -1 will on any axes will be translated to -1 in the final.

        Parameters:
            indices : a tuple of index per dimension.

            fullshape : a tuple of the shape of the full array

        Returns:
            ind : a 3-d array of the indices of the coordinates in indices in
                an array of size fullshape. -1 if any indices is -1.

    """
    localshape = [ len(i) for i in indices]
    ndim = len(localshape)
    ind = numpy.zeros(localshape, dtype='i8')
    for d in range(len(indices)):
        i = indices[d]
        i = i.reshape([-1 if dd == d else 1 for dd in range(ndim)])
        ind[...] *= fullshape[d]
        ind[...] += i

    # now mask out bad points by -1
    for d in range(len(indices)):
        i = indices[d]
        i = i.reshape([-1 if dd == d else 1 for dd in range(ndim)])
        ind[i == -1] = -1

    return ind

def reindex(Nsrc, Ndest):
    """ returns the index in the frequency array for corresponding
        k in Nsrc and composes Ndest

        For those Ndest that doesn't exist in Nsrc, return -1

        Example:
        >>> reindex(8, 4)
        >>> array([0, 1, 6, 7])
        >>> reindex(4, 8)
        >>> array([ 0,  1, -1, -1, -1, -1,  2,  3])

    """
    reindex = numpy.arange(Ndest)
    reindex[Ndest // 2:] = numpy.arange(Nsrc - Ndest // 2, Nsrc, 1)
    reindex[Nsrc // 2: -Nsrc //2] = -1
    return reindex
