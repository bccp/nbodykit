from nbodykit.extensionpoints import DataSource
from nbodykit.utils import selectionlanguage
import numpy
import bigfile

def ptypes_type(value):
    choices = ['0', '1', '2', '3', '4', '5', 'FOFGroups']
    if not all(v in choices for v in value):
        raise ValueError("valid choices for `ptypes` are: %s" %str(choices))
    return value
    
class BlueTidesDataSource(DataSource):
    plugin_name = "BlueTides"
    
    def __init__(self, path, BoxSize, ptypes=None, load=[], 
                    subsample=False, bunchsize=4*1024*1024, select=None):
        pass
    
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "read from the BlueTides simulation"
        
        s.add_argument("path", help="path to file")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("ptypes", type=ptypes_type, help="type of particle to read")
        s.add_argument("load", type=list, help="extra columns to load")
        s.add_argument("subsample", type=bool, help="this is a subsample file")
        s.add_argument("bunchsize", type=int, help="number of particle to read in a bunch")
        s.add_argument("-select", type=selectionlanguage.Query,
            help='row selection e.g. Mass > 1e3 and Mass < 1e5')
    
    def parallel_read(self, columns, full=False):
         
        f = bigfile.BigFile(self.path)
        header = f['header']
        boxsize = header.attrs['BoxSize'][0]

        ptypes = self.ptypes
        readcolumns = []
        for column in columns:
            if column == 'HI':
                if 'Mass' not in readcolumns:
                    readcolumns.append('Mass')
                if 'NeutralHydrogenFraction' not in readcolumns:
                    readcolumns.append('NeutralHydrogenFraction')
            else:
                readcolumns.append(column)

        readcolumns = readcolumns + self.load
        for ptype in ptypes:
            for data in self.read_ptype(ptype, readcolumns, full):
                P = dict(zip(readcolumns, data))
                if 'HI' in columns:
                    P['HI'] = P['NeutralHydrogenFraction'] * P['Mass']

                if 'Position' in columns:
                    P['Position'][:] *= self.BoxSize / boxsize
                    P['Position'][:] %= self.BoxSize

                if 'Velocity' in columns:
                    raise NotImplementedError

                if self.select is not None:
                    mask = self.select.get_mask(P)
                else:
                    mask = Ellipsis
                toret = []
                for column in columns:
                    d = P.get(column, None)
                    if d is not None:
                        d = d[mask]
                    toret.append(d)
                yield toret

    def read_ptype(self, ptype, columns, full):
        f = bigfile.BigFile(self.path)
        done = False
        i = 0
        while not numpy.all(self.comm.allgather(done)):
            ret = []
            for column in columns:
                f = bigfile.BigFile(self.path)
                read_column = column
                if self.subsample:
                    if ptype in ("0", "1"):
                        read_column = read_column + '.sample'

                if ptype == 'FOFGroups':
                    if column == 'Position':
                        read_column = 'MassCenterPosition'
                    if column == 'Velocity':
                        read_column = 'MassCenterVelocity'

                cdata = f['%s/%s' % (ptype, read_column)]

                Ntot = cdata.size
                start = self.comm.rank * Ntot // self.comm.size
                end = (self.comm.rank + 1) * Ntot //self.comm.size
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
                data = cdata[bunchstart:bunchend]
                ret.append(data)
            i = i + 1
            yield ret

