from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser
from nbodykit import files 
import logging

#------------------------------------------------------------------------------
class TPMSnapshotDataSource(DataSource):
    field_type = "TPMSnapshot"
    
    @classmethod
    def register(kls):
        
        h = kls.add_parser()
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
    def read(self, columns, comm, bunchsize):
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

        for round, P in enumerate(
                files.read(comm, 
                    self.path, 
                    files.TPMSnapshotFile, 
                    columns=newcolumns, 
                    bunchsize=bunchsize)):

            if comm.rank == 0:
                logging.info('round %d, nread %d' % (round, len(P['Position'])))

            P['Position'] *= self.BoxSize
            P['Mass'] = None
            if 'Velocity' in P:
                P['Velocity'] *= self.BoxSize

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= self.BoxSize[dir]

            for key in P:
                if key not in columns:
                    del P[key]

            yield P

#------------------------------------------------------------------------------
