from nbodykit.core import DataSource
from nbodykit import files 
import numpy

class TPMSnapshotDataSource(DataSource):
    """
    DataSource to read snapshot files from Martin White's TPM simulations
    """
    plugin_name = "TPMSnapshot"
    
    def __init__(self, path, BoxSize, rsd=None, bunchsize=4*1024*1024):
        
        self.path = path
        self.BoxSize = BoxSize
        self.rsd = rsd
        self.bunchsize = bunchsize
        
        if self.comm.rank == 0:
            datastorage = files.DataStorage(self.path, files.TPMSnapshotFile)
            size = sum(datastorage.npart)
        else:
            size = None
        self.size = self.comm.bcast(size)

    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "read snapshot files from Martin White's TPM"
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("rsd", choices="xyz", help="direction to do redshift distortion")
        s.add_argument("bunchsize", type=int, help="number of particles to read per rank in a bunch")

    def parallel_read(self, columns, full=False):
        """ 
        read data in parallel. if Full is True, neglect bunchsize.
        
        This supports `Position`, `Velocity` columns
        """
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

        if self.comm.rank == 0:
            datastorage = files.DataStorage(self.path, files.TPMSnapshotFile)
        else:
            datastorage = None
        datastorage = self.comm.bcast(datastorage)

        for round, P0 in enumerate(
                datastorage.iter(comm=self.comm, 
                    columns=newcolumns, bunchsize=bunchsize)):
            P = dict(zip(newcolumns, P0))
            if 'Position' in P:
                P['Position'] *= self.BoxSize
            if 'Velocity' in P:
                P['Velocity'] *= self.BoxSize

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= self.BoxSize[dir]

            yield [P.get(key, None) for key in columns]

