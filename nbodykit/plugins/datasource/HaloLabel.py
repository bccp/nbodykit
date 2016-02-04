from nbodykit.extensionpoints import DataSource
import numpy
import logging
import bigfile

logger = logging.getLogger('FastPM')

class HaloLabel(DataSource):
    plugin_name = "HaloLabel"
    
    @classmethod
    def register(kls):
        h = kls.parser

        h.add_argument("path", help="path to file")
        h.add_argument("-bunchsize", type=int, default=4 *1024*1024,
                help="number of particle to read in a bunch")
    
    def read(self, columns, stats, full=False):
        f = bigfile.BigFileMPI(self.comm, self.path)
        readcolumns = columns
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

            stats['Ntot'] += self.comm.allreduce(bunchend - bunchstart)
            i = i + 1
            yield [P[column] for column in columns]
