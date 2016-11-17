from nbodykit.core import Algorithm, DataSource, Painter
import numpy

class TidalTensor(Algorithm):
    """
    Compute and save the tidal force tensor
    """
    plugin_name = "TidalTensor"
    
    def __init__(self, field, points, Nmesh, smoothing=None):
        from pmesh.pm import ParticleMesh
        
        self.field     = field
        self.points    = points
        self.Nmesh     = Nmesh
        self.smoothing = smoothing
        
        self.pm = ParticleMesh(BoxSize=self.field.BoxSize, Nmesh=[self.Nmesh]*3, dtype='f4', comm=self.comm)

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "compute the tidal force tensor"

        s.add_argument("field", type=DataSource.from_config,
                help="DataSource; run `nbkit.py --list-datasources` for all options")
        s.add_argument("points", type=DataSource.from_config,
                help="A small set of points to calculate tidal force on; "
                     "run `nbkit.py --list-datasources` for all options")
        s.add_argument("Nmesh", type=int,
                help='Size of FFT mesh for painting')
        s.add_argument("smoothing", type=float,
                help='Smoothing Length in distance units. '
                      'It has to be greater than the mesh resolution. '
                      'Otherwise the code will die. Default is the mesh resolution.')

    def Smoothing(self, pm, complex):
        k = pm.k
        k2 = 0
        for ki in k:
            ki2 = ki ** 2
            complex[:] *= numpy.exp(-0.5 * ki2 * self.smoothing ** 2)

    def NormalizeDC(self, pm, complex):
        """ removes the DC amplitude. This effectively
            divides by the mean
        """

        w = pm.w
        comm = pm.comm
        ind = []
        value = 0.0
        found = True
        for wi in w:
            if (wi != 0).all():
                found = False
                break
            ind.append((wi == 0).nonzero()[0][0])
        if found:
            ind = tuple(ind)
            value = numpy.abs(complex[ind])
        value = comm.allreduce(value)
        complex[:] /= value

    def TidalTensor(self, u, v):
        # k_u k_v / k **2
        def TidalTensor(pm, complex):
            # iterate over slabs
            for kk, slab in zip(complex.slabs.x, complex.slabs):
            
                k2 = kk[0]**2 + kk[1]**2 + kk[2]**2
                k2[k2 == 0] = numpy.inf
                slab[:] /= k2
                
                slab[:] *= kk[u]
                slab[:] *= kk[v]

        return TidalTensor

    def run(self):
        """
        Run the TidalTensor Algorithm
        """
        from itertools import product
        
        # determine smoothing
        if self.smoothing is None:
            self.smoothing = self.field.BoxSize[0] / self.Nmesh
        elif (self.field.BoxSize / self.Nmesh > self.smoothing).any():
            raise ValueError("smoothing is too small")
            
        # paint the field and FFT
        painter = Painter.create("DefaultPainter", weight="Mass", paintbrush="cic")
        real, stats = painter.paint(self.pm, self.field)
        complex = real.r2c()

        # apply transfers
        for t in [self.Smoothing, self.NormalizeDC]:
            t(complex.pm, complex)

        # read the points
        with self.points.open() as stream:
            [[Position]] = stream.read(['Position'], full=True)

        layout = self.pm.decompose(Position)
        pos1 = layout.exchange(Position)
        value = numpy.empty((3, 3, len(Position)))

        # do tidal tensor calculation
        for u, v in product(range(3), range(3)):
            if self.comm.rank == 0:
                self.logger.info("Working on tensor element (%d, %d)" % (u, v))
            c2 = complex.copy()

            self.TidalTensor(u, v)(c2.pm, c2)
            c2.c2r(real)
            v1 = real.readout(pos1)
            v1 = layout.gather(v1)

            value[u, v] = v1

        return value.transpose((2, 0, 1))

    def save(self, output, data):
        self.write_hdf5(data, output)

    def write_hdf5(self, data, output):
        import h5py
        size = self.comm.allreduce(len(data))
        offset = sum(self.comm.allgather(len(data))[:self.comm.rank])

        if self.comm.rank == 0:
            with h5py.File(output, 'w') as ff:
                dataset = ff.create_dataset(name='TidalTensor',
                        dtype=data.dtype, shape=(size, 3, 3))
                dataset.attrs['Smoothing'] = self.smoothing
                dataset.attrs['Nmesh'] = self.Nmesh
                dataset.attrs['Original'] = self.field.string
                dataset.attrs['BoxSize'] = self.field.BoxSize

        for i in range(self.comm.size):
            self.comm.barrier()
            if i != self.comm.rank: continue

            with h5py.File(output, 'r+') as ff:
                dataset = ff['TidalTensor']
                dataset[offset:len(data) + offset] = data

