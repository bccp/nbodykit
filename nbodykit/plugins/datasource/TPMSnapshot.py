from nbodykit.extensionpoints import DataSource
from nbodykit import files 
import numpy

class TPMSnapshotDataSource(DataSource):
    field_type = "TPMSnapshot"
    
    @classmethod
    def register(kls):
        
        h = kls.parser
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=kls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
        h.add_argument("-bunchsize", type=int, 
                default=1024*1024*4, help="number of particles to read per rank in a bunch")

    def read(self, columns, comm, stats, full=False):
        """ read data in parallel. if Full is True, neglect bunchsize. """
        Ntot = 0
        # avoid reading Velocity if RSD is not requested.
        # this is only needed for large data like a TPMSnapshot
        # for small Pandas reader etc it doesn't take time to
        # read velocity

        if self.rsd is not None:
            newcolumns = set(columns + ['Velocity'])
        else:
            newcolumns = set(columns)

        if 'Mass' in newcolumns:
            newcolumns.remove('Mass')
        if 'Weight' in newcolumns:
            newcolumns.remove('Weight')

        bunchsize = self.bunchsize
        if full: bunchsize = None
        stats['Ntot'] = 0

        for round, P in enumerate(
                files.read(comm, 
                    self.path, 
                    files.TPMSnapshotFile, 
                    columns=newcolumns, 
                    bunchsize=bunchsize)):

            if 'Position' in P:
                P['Position'] *= self.BoxSize
            if 'Velocity' in P:
                P['Velocity'] *= self.BoxSize

            # uniform mass
            P['Mass'] = numpy.ones(P['__nread__'], 'i1')
            P['Weight'] = P['Mass']

            stats['Ntot'] += comm.allreduce(P['__nread__'])

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= self.BoxSize[dir]

            yield [P[key] for key in columns]

#------------------------------------------------------------------------------
