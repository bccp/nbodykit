from nbodykit.plugins import DataSource
import numpy
import logging
import bigfile

logger = logging.getLogger('BlueTides')

class BlueTidesDataSource(DataSource):
    field_type = "BlueTides"
    @classmethod
    def register(kls):
        
        h = kls.add_parser()
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=kls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-ptype", type=int,
            choices=[0, 1, 2, 3, 4, 5], help="type of particle to read")
        h.add_argument("-subsample", type=bool, help="this is a subsample file")
    
    def read(self, columns, comm, full=False):
        
        ptypes = [self.ptype]
        for ptype in ptypes:
            for data in self.read_ptype(ptype, columns, comm, full):
                yield data

    def read_ptype(self, ptype, columns, comm, full):
        ret = []
        f = bigfile.BigFile(self.path)
        header = f['header']
        boxsize = header.attrs['BoxSize'][0]

        for column in columns:
            f = bigfile.BigFile(self.path)
            read_column = column
            if column == 'Weight': read_column = 'Mass'
            if self.subsample:
                if ptype in (0, 1):
                    read_column = read_column + '.sample'
            cdata = f['%d/%s' % (self.ptype, read_column)]

            Ntot = cdata.size
            start = comm.rank * Ntot // comm.size
            end = (comm.rank + 1) * Ntot //comm.size
            data = cdata[start:end]
            ret.append(data)
            if column == 'Position':
                data *= self.BoxSize / boxsize
            if column == 'Velocity':
                raise NotImplementedError
        yield ret
