from nbodykit.extensionpoints import DataSource
import numpy
import logging
import bigfile

logger = logging.getLogger('FastPM')

class FastPMDataSource(DataSource):
    plugin_name = "FastPM"
    
    @classmethod
    def register(kls):
        h = kls.parser

        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=kls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-bunchsize", type=int, default=4 *1024*1024,
                help="number of particle to read in a bunch")
        h.add_argument("-rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
    
    def read(self, columns, comm, stats, full=False):
        f = bigfile.BigFile(self.path)
        header = f['header']
        boxsize = header.attrs['BoxSize'][0]
        RSD = header.attrs['RSDFactor'][0]
        if boxsize != self.BoxSize[0]:
            raise ValueError("Box size mismatch, expecting %g" % boxsize)

        readcolumns = columns
        if self.rsd is not None:
            readcolumns = set(columns + ['Velocity'])

        stats['Ntot'] = 0
        done = False
        i = 0
        while not numpy.all(comm.allgather(done)):
            ret = []
            dataset = bigfile.BigData(f, readcolumns)

            Ntot = dataset.size
            start = comm.rank * Ntot // comm.size
            end = (comm.rank + 1) * Ntot //comm.size

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

            if 'Mass' in readcolumns:
                P['Mass'] = numpy.ones(stats['Ncurrent'], dtype='u1')

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= self.BoxSize[dir]

            stats['Ntot'] += comm.allreduce(bunchend - bunchstart)
            i = i + 1
            yield [P[column] for column in columns]
