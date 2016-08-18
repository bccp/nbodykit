from nbodykit.core import Algorithm, DataSource, Painter
from nbodykit import utils
import numpy

class Subsample(Algorithm):
    """
    Algorithm to create a subsample from a DataSource, and evaluate
    the density (1 + delta), smoothed at the given scale
    """
    plugin_name = "Subsample"
    
    def __init__(self, datasource, Nmesh, seed=12345, ratio=0.01, smoothing=None, format='hdf5'):
        from pmesh.pm import ParticleMesh

        self.datasource = datasource
        self.Nmesh      = Nmesh
        self.seed       = seed
        self.ratio      = ratio
        self.smoothing  = smoothing
        self.format     = format

        self.pm = ParticleMesh(BoxSize=self.datasource.BoxSize, Nmesh=[self.Nmesh] * 3, dtype='f4', comm=self.comm)

    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "create a subsample from a DataSource, and evaluate density \n"
        s.description += "(1 + delta) smoothed at the given scale"
        
        s.add_argument("datasource", type=DataSource.from_config,
            help="the DataSource to read; run `nbkit.py --list-datasources` for all options")
        s.add_argument("Nmesh", type=int, help='the size of FFT mesh for painting')
        s.add_argument("seed", type=int, help='the random seed')
        s.add_argument("ratio", type=float, help='the fraction of particles to keep')
        s.add_argument("smoothing", type=float,
                help='the smoothing length in distance units. '
                      'It has to be greater than the mesh resolution. '
                      'Otherwise the code will die. Default is the mesh resolution.')
        # this is for output..
        s.add_argument("format", choices=['hdf5', 'mwhite'], help='the format of the output')

    def run(self):
        """
        Run the Subsample algorithm
        """
        import mpsort
        from astropy.utils.misc import NumpyRNGContext

        if self.smoothing is None:
            self.smoothing = self.datasource.BoxSize[0] / self.Nmesh[0]
        elif (self.datasource.BoxSize / self.Nmesh > self.smoothing).any():
            raise ValueError("smoothing is too small")
     
        def Smoothing(pm, complex):
            k = pm.k
            k2 = 0
            for ki in k:
                ki2 = ki ** 2
                complex[:] *= numpy.exp(-0.5 * ki2 * self.smoothing ** 2)

        def NormalizeDC(pm, complex):
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

        # open the datasource and keep the cache
        with self.datasource.keep_cache():

            painter = Painter.create("DefaultPainter", paintbrush='cic')
            real, stats = painter.paint(self.pm, self.datasource)
            complex = real.r2c()

            for t in [Smoothing, NormalizeDC]: t(self.pm, complex)

            complex.c2r(real)

            columns = ['Position', 'ID', 'Velocity']
            local_seed = utils.local_random_seed(self.seed, self.comm)

            dtype = numpy.dtype([
                    ('Position', ('f4', 3)),
                    ('Velocity', ('f4', 3)),
                    ('ID', 'u8'),
                    ('Density', 'f4'),
                    ])

            subsample = [numpy.empty(0, dtype=dtype)]

            with self.datasource.open() as stream:
                for Position, ID, Velocity in stream.read(columns):

                    with NumpyRNGContext(local_seed):
                        u = numpy.random.uniform(size=len(ID))
                    keep = u < self.ratio
                    Nkeep = keep.sum()
                    if Nkeep == 0: continue 
                    data = numpy.empty(Nkeep, dtype=dtype)
                    data['Position'][:] = Position[keep]
                    data['Velocity'][:] = Velocity[keep]
                    data['ID'][:] = ID[keep]

                    layout = self.pm.decompose(data['Position'])
                    pos1 = layout.exchange(data['Position'])
                    density1 = real.readout(pos1)
                    density = layout.gather(density1)

                    # normalize the position after reading out density!
                    data['Position'][:] /= self.datasource.BoxSize
                    data['Velocity'][:] /= self.datasource.BoxSize
                    data['Density'][:] = density
                    subsample.append(data)

        subsample = numpy.concatenate(subsample)
        mpsort.sort(subsample, orderby='ID', comm=self.comm)

        return subsample

    def save(self, output, data):
        if self.format == 'mwhite':
            self.write_mwhite_subsample(data, output)
        else:
            self.write_hdf5(data, output)

    def write_hdf5(self, subsample, output):
        import h5py
        
        size = self.comm.allreduce(len(subsample))
        offset = sum(self.comm.allgather(len(subsample))[:self.comm.rank])

        if self.comm.rank == 0:
            with h5py.File(output, 'w') as ff:
                dataset = ff.create_dataset(name='Subsample',
                        dtype=subsample.dtype, shape=(size,))
                dataset.attrs['Ratio'] = self.ratio
                dataset.attrs['CommSize'] = self.comm.size 
                dataset.attrs['Seed'] = self.seed
                dataset.attrs['Smoothing'] = self.smoothing
                dataset.attrs['Nmesh'] = self.Nmesh
                dataset.attrs['Original'] = self.datasource.string
                dataset.attrs['BoxSize'] = self.datasource.BoxSize

        for i in range(self.comm.size):
            self.comm.barrier()
            if i != self.comm.rank: continue
                 
            with h5py.File(output, 'r+') as ff:
                dataset = ff['Subsample']
                dataset[offset:len(subsample) + offset] = subsample

    def write_mwhite_subsample(self, subsample, output):
        size = self.comm.allreduce(len(subsample))
        offset = sum(self.comm.allgather(len(subsample))[:self.comm.rank])

        if self.comm.rank == 0:
            with open(output, 'wb') as ff:
                dtype = numpy.dtype([
                        ('eflag', 'int32'),
                        ('hsize', 'int32'),
                        ('npart', 'int32'),
                         ('nsph', 'int32'),
                         ('nstar', 'int32'),
                         ('aa', 'float'),
                         ('gravsmooth', 'float')])
                header = numpy.zeros((), dtype=dtype)
                header['eflag'] = 1
                header['hsize'] = 20
                header['npart'] = size
                header.tofile(ff)

        self.comm.barrier()

        with open(output, 'r+b') as ff:
            ff.seek(28 + offset * 12)
            numpy.float32(subsample['Position']).tofile(ff)
            ff.seek(28 + offset * 12 + size * 12)
            numpy.float32(subsample['Velocity']).tofile(ff)
            ff.seek(28 + offset * 4 + size * 24)
            numpy.float32(subsample['Density']).tofile(ff)
            ff.seek(28 + offset * 8 + size * 28)
            numpy.uint64(subsample['ID']).tofile(ff)
