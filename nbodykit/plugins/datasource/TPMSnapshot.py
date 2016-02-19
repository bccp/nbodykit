from nbodykit.extensionpoints import DataSource
from nbodykit import files 
import numpy

class TPMSnapshotDataSource(DataSource):
    """
    DataSource to read snapshot files from Martin White's TPM simulations
    """
    plugin_name = "TPMSnapshot"
    
    def __init__(self, path, BoxSize, rsd=None, bunchsize=4*1024*1024):
        pass

    @classmethod
    def register(cls):
        
        s = cls.schema
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("rsd", choices="xyz", help="direction to do redshift distortion")
        s.add_argument("bunchsize", type=int, help="number of particles to read per rank in a bunch")

    def read(self, columns, stats, full=False):
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
        if full: bunchsize = -1

        stats['Ntot'] = 0
        if self.comm.rank == 0:
            datastorage = files.DataStorage(self.path, files.TPMSnapshotFile)
        else:
            datastorage = None
        datastorage = self.comm.bcast(datastorage)

        for round, P in enumerate(
                datastorage.iter(stats=stats, comm=self.comm, 
                    columns=newcolumns, bunchsize=bunchsize)):
            P = dict(zip(newcolumns, P))
            if 'Position' in P:
                P['Position'] *= self.BoxSize
            if 'Velocity' in P:
                P['Velocity'] *= self.BoxSize

            P['Mass'] = numpy.ones(stats['Ncurrent'], dtype='u1')
            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= self.BoxSize[dir]

            yield [P[key] for key in columns]

#------------------------------------------------------------------------------
