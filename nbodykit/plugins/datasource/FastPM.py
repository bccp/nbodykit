from nbodykit.extensionpoints import DataSource
import numpy
import logging
import bigfile

logger = logging.getLogger('FastPM')

class FastPMDataSource(DataSource):
    """
    DataSource to read snapshot files of the FastPM simulation
    """
    plugin_name = "FastPM"
    
    def __init__(self, path, BoxSize=None, bunchsize=4*1024*1024, rsd=None):
        
        BoxSize = numpy.empty(3, dtype='f8')
        f = bigfile.BigFileMPI(self.comm, self.path)
        header = f['header']
        BoxSize[:] = header.attrs['BoxSize'][0]
        OmegaM = header.attrs['OmegaM'][0]
        self.M0 = 27.75e10 * OmegaM * BoxSize[0] ** 3 / f['Position'].size
        self.size = f['Position'].size

        if self.comm.rank == 0:
            logger.info("File has boxsize of %s Mpc/h" % str(BoxSize))
            logger.info("Mass of a particle is %g Msun/h" % self.M0)

        if self.BoxSize is None:
            self.BoxSize = BoxSize
        else:
            if self.comm.rank == 0:
                logger.info("Overriding boxsize as %s" % str(self.BoxSize))
        
    
    @classmethod
    def register(cls):
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
                
    def parallel_read(self, columns, full=False):
        f = bigfile.BigFileMPI(self.comm, self.path)
        header = f['header']
        boxsize = header.attrs['BoxSize'][0]
        RSD = header.attrs['RSDFactor'][0]
        if boxsize != self.BoxSize[0]:
            raise ValueError("Box size mismatch, expecting %g" % boxsize)

        readcolumns = set(columns)
        if self.rsd is not None:
            readcolumns = set(columns + ['Velocity'])
        if 'InitialPosition' in columns:
            readcolumns.add('ID')
            readcolumns.remove('InitialPosition')

        if 'Mass' in readcolumns: 
            readcolumns.remove('Mass')
            
        # remove columns not in the file (None will be returned)
        for col in list(readcolumns):
            if col not in f:
                readcolumns.remove(col)
            
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
                P['Velocity'] *= RSD

            if 'Mass' in columns:
                P['Mass'] = numpy.ones(bunchend - bunchstart, dtype='u1') * self.M0

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
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
