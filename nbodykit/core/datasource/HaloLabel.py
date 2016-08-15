from nbodykit.core import DataSource
import numpy
import bigfile

class HaloLabel(DataSource):
    """
    DataSource for reading a file of halo labels (halo id per particle), 
    as generated the FOF algorithm
    """
    plugin_name = "HaloLabel"
    
    def __init__(self, path, bunchsize=4*1024*1024):
        
        self.path = path
        self.bunchsize = bunchsize 
        
        f = bigfile.BigFileMPI(self.comm, self.path)
        self.size = f['Label'].size
    
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "read a file of halo labels (halo id per particle), as generated the FOF algorithm"
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("bunchsize", type=int, help="number of particle to read in a bunch")

    def parallel_read(self, columns, full=False):
        f = bigfile.BigFileMPI(self.comm, self.path)
        readcolumns = set(columns)
        
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

            i = i + 1
            yield [P.get(column, None) for column in columns]
