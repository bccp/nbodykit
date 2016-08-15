from nbodykit.core import DataSource
from nbodykit import files 
import numpy

class TPMLabel(DataSource):
    """
    DataSource to read file of halo labels (halo id per particle), 
    as generated from Martin White's TPM simulations
    """
    plugin_name = "TPMLabel"
    
    def __init__(self, path, bunchsize=4*1024*1024):
        
        self.path      = path
        self.bunchsize = bunchsize
        
        if self.comm.rank == 0:
            datastorage = files.DataStorage(self.path, files.HaloLabelFile)
        else:
            datastorage = None
        datastorage = self.comm.bcast(datastorage)
        self.size = sum(datastorage.npart)
    
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "read file of halo labels as generated from Martin White's TPM"
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("bunchsize", type=int, help="number of particle to read in a bunch")

    def parallel_read(self, columns, full=False):
        """ read data in parallel. if Full is True, neglect bunchsize. """
        Ntot = 0
        # avoid reading Velocity if RSD is not requested.
        # this is only needed for large data like a TPMSnapshot
        # for small Pandas reader etc it doesn't take time to
        # read velocity

        bunchsize = self.bunchsize
        if full: bunchsize = -1

        if self.comm.rank == 0:
            datastorage = files.DataStorage(self.path, files.HaloLabelFile)
        else:
            datastorage = None
        datastorage = self.comm.bcast(datastorage)

        for round, P in enumerate(
                datastorage.iter(comm=self.comm, 
                    columns=columns, bunchsize=bunchsize)):
            P = dict(zip(columns, P))

            yield [P.get(key, None) for key in columns]
