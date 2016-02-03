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
        h.add_argument("-BoxSize", type=kls.BoxSizeParser, default=None,
            help="Override the size of the box, can be a scalar or a three vector")
        h.add_argument("-bunchsize", type=int, default=4 *1024*1024,
                help="number of particle to read in a bunch")
        h.add_argument("-rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
    
    def finalize_attributes(self):
        BoxSize = numpy.empty(3, dtype='f8')
        if self.comm.rank == 0:
            f = bigfile.BigFile(self.path)
            header = f['header']
            BoxSize[:] = header.attrs['BoxSize'][0]
            logger.info("File has boxsize of %s" % str(BoxSize))
        BoxSize = self.comm.bcast(BoxSize)
        if self.BoxSize is None:
            self.BoxSize = BoxSize
        else:
            if self.comm.rank == 0:
                logger.info("Overriding boxsize as %s" % str(self.BoxSize))
    
    def read(self, columns, stats, full=False):
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

            if 'Mass' in readcolumns:
                P['Mass'] = numpy.ones(stats['Ncurrent'], dtype='u1')

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= self.BoxSize[dir]

            stats['Ntot'] += self.comm.allreduce(bunchend - bunchstart)
            i = i + 1
            yield [P[column] for column in columns]
