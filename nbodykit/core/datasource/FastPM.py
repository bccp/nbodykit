from nbodykit.core import DataSource
import numpy
import bigfile

def interpMake(f, xmin, xmax, steps):
        xi = []
        fi = []
        delta = (xmax - xmin)/steps

        for i in range(0, steps):
            t = xmin + i * delta
            fi.append(f(t).value)
            xi.append(t)

        def finterp(x):
            return numpy.interp(x, xi, fi)
        return finterp

class FastPMDataSource(DataSource):
    """
    DataSource to read snapshot files of the FastPM simulation
    """
    plugin_name = "FastPM"

    def __init__(self, path, BoxSize=None, bunchsize=4*1024*1024, rsd=None, lightcone=False, potentialRSD=False, velocityRSD=True):

        self.path      = path
        self.BoxSize   = BoxSize
        self.bunchsize = bunchsize
        self.rsd       = rsd
        self.lightcone = lightcone
        self.potentialRSD = potentialRSD
        self.velocityRSD = velocityRSD

        BoxSize = numpy.empty(3, dtype='f8')
        f = bigfile.BigFileMPI(self.comm, self.path)
        try:
            header = f['header']
        except:
            try:
                header = f['Header']
            except:
                header = f['.']

        try:
            f = f['1/']
        except:
            pass

        BoxSize[:] = header.attrs['BoxSize'][0]
        OmegaM = header.attrs['OmegaM'][0]
        self.M0 = 27.75e10 * OmegaM * BoxSize[0] ** 3 / f['Position'].size
        self.size = f['Position'].size

        if self.comm.rank == 0:
            self.logger.info("File has boxsize of %s Mpc/h" % str(BoxSize))
            self.logger.info("Mass of a particle is %g Msun/h" % self.M0)

        if self.BoxSize is None:
            self.BoxSize = BoxSize
        else:
            if self.comm.rank == 0:
                self.logger.info("Overriding boxsize as %s" % str(self.BoxSize))

    @classmethod
    def fill_schema(cls):
        """
        Fill the attribute schema associated with this class
        """
        s = cls.schema
        s.description = "read snapshot files of the FastPM simulation"

        s.add_argument("path", type=str,
            help="the file path to load the data from")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="override the size of the box; can be a scalar or a three vector")
        s.add_argument("rsd", type=str, choices="xyz",
            help="direction to do redshift distortion")
        s.add_argument("bunchsize", type=int,
            help="number of particles to read per rank in a bunch")
        s.add_argument("lightcone", type=bool, help="potential displacement for lightcone")
        s.add_argument("potentialRSD", type=bool, help="potential included in file")
        s.add_argument("velocityRSD", type=bool, help="velocity included in file")

    def parallel_read(self, columns, full=False):
        f = bigfile.BigFileMPI(self.comm, self.path)
        try:
            header = f['header']
        except:
            try:
                header = f['Header']
            except:
                header = f['.']

        boxsize = header.attrs['BoxSize'][0]
        RSD = header.attrs['RSDFactor'][0]
        if boxsize != self.BoxSize[0]:
            raise ValueError("Box size mismatch, expecting %g" % boxsize)

        try:
            f = f['1/']
        except:
            pass

        readcolumns = set(columns)
        if self.rsd is not None:
            readcolumns = set(columns + ['Velocity'])
        if self.lightcone:
            if self.potentialRSD:
                readcolumns = set(columns + ['Velocity', 'Potential', 'Aemit'])
            else:
                readcolumns = set(columns + ['Velocity', 'Aemit'])
        if 'InitialPosition' in columns:
            readcolumns.add('ID')
            readcolumns.remove('InitialPosition')

        if 'Mass' in readcolumns:
            readcolumns.remove('Mass')

        # remove columns not in the file (None will be returned)
        for col in list(readcolumns):
            if col not in f:
                readcolumns.remove(col)

        # sort to the correct order (order is non-deterministic!!)
        readcolumns = sorted(readcolumns)

        done = False
        i = 0
        while not numpy.all(self.comm.allgather(done)):
            ret = []
            dataset = bigfile.BigData(f, readcolumns)

            Ntot = dataset.size
            start = self.comm.rank * Ntot // self.comm.size
            end = (self.comm.rank + 1) * Ntot // self.comm.size

            if not full:
                bunchstart = start + i * self.bunchsize
                bunchend = start + (i + 1) * self.bunchsize
                if bunchend > end: bunchend = end
                if bunchstart > end: bunchstart = end
            else:
                bunchstart = start
                bunchend = end

            if bunchend == end:
                done = True

            P = {}

            for column in readcolumns:
                data = dataset[column][bunchstart:bunchend]
                P[column] = data

            if 'Velocity' in P:
                if not self.lightcone:
                    P['Velocity'] *= RSD
                else:
                    #H = self.cosmo.engine.H(redshift) / self.cosmo.engine.h

                    H_interp = interpMake(self.cosmo.engine.H, 0, 20, 8192) # bounds from 0 to 20 with 8000 steps

                    redshift = 1/(P['Aemit']) - 1
                    H = H_interp(redshift)/self.cosmo.engine.h
                    factor = 1./(P['Aemit']*H)

                    P['Velocity'] *= factor[:, None]

                    if self.potentialRSD:
                        P['Potential']*= factor*3*(10**5)

            if 'Mass' in columns:
                P['Mass'] = numpy.ones(bunchend - bunchstart, dtype='u1') * self.M0

            if 'Position' in columns:
                if self.rsd is not None:
                    dir = "xyz".index(self.rsd)
                    if self.velocityRSD:
                        P['Position'][:, dir] += P['Velocity'][:, dir]
                    if self.potentialRSD:
                        P['Position'][:, dir] += P['Potential']
                    P['Position'][:, dir] %= self.BoxSize[dir]

            if 'InitialPosition' in columns:
                P['InitialPosition'] = numpy.empty((len(P['ID']), 3), 'f4')
                nc = int(self.size ** (1. / 3) + 0.5)
                id = P['ID'].copy()
                for nc in range(nc - 10, nc + 10):
                    if nc ** 3 == self.size: break
                for d in [2, 1, 0]:
                    P['InitialPosition'][:, d] = id % nc
                    id[:] //= nc
                cellsize = self.BoxSize[0] / nc
                P['InitialPosition'][:] += 0.5
                P['InitialPosition'][:] *= cellsize

            i = i + 1
            yield [P.get(column, None) for column in columns]
