from nbodykit.core import DataSource
from nbodykit import files 
import numpy

class GadgetDataSource(DataSource):
    """
    A DataSource to read a flavor of Gadget 2 files (experimental)
    """
    plugin_name = "Gadget"
    
    def __init__(self, path, BoxSize, ptype=[], posdtype='f4', veldtype='f4', 
                    iddtype='u8', massdtype='f4', rsd=None, bunchsize=4*1024*1024):
        
        # positional arguments
        self.path = path
        self.BoxSize = BoxSize
        
        # keywords
        self.ptype     = ptype
        self.posdtype  = posdtype
        self.veldtype  = veldtype
        self.iddtype   = iddtype
        self.massdtype = massdtype
        self.rsd       = rsd
        self.bunchsize = bunchsize
        
    
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "read a flavor of Gadget 2 files (experimental)"
        
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("ptype", nargs='*', type=int, help="the particle types to use")
        s.add_argument("posdtype", type=str, help="dtype of position")
        s.add_argument("veldtype", type=str, help="dtype of velocity")
        s.add_argument("iddtype", type=str, help="dtype of id")
        s.add_argument("massdtype", type=str, help="dtype of mass")
        s.add_argument("rsd", choices="xyz", type=str, help="direction to do redshift distortion")
        s.add_argument("bunchsize", type=int, help="number of particles to read per rank in a bunch")

    def parallel_read(self, columns, full=False):
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

        bunchsize = self.bunchsize
        if full: bunchsize = -1
        if full and len(self.ptype) > 1:
            raise ValueError("cannot read multple ptype in a full load")
        for ptype in self.ptype:
            args = dict(ptype=ptype,
                 posdtype=self.posdtype,
                 veldtype=self.veldtype,
                 massdtype=self.massdtype,
                 iddtype=self.iddtype)

            if self.comm.rank == 0:
                datastorage = files.DataStorage(self.path,
                        files.GadgetSnapshotFile, args)
                f0 = files.GadgetSnapshotFile(self.path, 0, args)
                boxsize = f0.header['boxsize']
            else:
                datastorage = None
                boxsize = None
            boxsize = self.comm.bcast(boxsize)
            datastorage = self.comm.bcast(datastorage)

            for round, P in enumerate(
                    datastorage.iter(comm=self.comm, 
                        columns=newcolumns, bunchsize=bunchsize)):
                P = dict(zip(newcolumns, P))
                if 'Position' in P:
                    P['Position'] /= boxsize
                    P['Position'] *= self.BoxSize
                if 'Velocity' in P:
                    raise KeyError('Velocity is not yet supported')

                if self.rsd is not None:
                    dir = "xyz".index(self.rsd)
                    P['Position'][:, dir] += P['Velocity'][:, dir]
                    P['Position'][:, dir] %= self.BoxSize[dir]

                yield [P.get(key, None) for key in columns]


class GadgetGroupTabDataSource(DataSource):
    """
    A DataSource to read a flavor of Gadget 2 FOF catalogs (experimental)
    """
    plugin_name = "GadgetGroupTab"
    
    def __init__(self, path, BoxSize, mpch=1000., posdtype='f4', veldtype='f4', 
                    massdtype='f4', rsd=None, bunchsize=4*1024*1024):
                    
        # positional arguments
        self.path = path
        self.BoxSize = BoxSize
        
        # keywords
        self.mpch      = mpch
        self.posdtype  = posdtype
        self.veldtype  = veldtype
        self.massdtype = massdtype
        self.rsd       = rsd
        self.bunchsize = bunchsize
    
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "read a flavor of Gadget 2 FOF catalogs (experimental)"
        
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("mpch", type=float, help="Mpc/h in code unit")
        s.add_argument("posdtype", help="dtype of position")
        s.add_argument("veldtype", help="dtype of velocity")
        s.add_argument("massdtype", help="dtype of mass")
        s.add_argument("rsd", choices="xyz", help="direction to do redshift distortion")
        s.add_argument("bunchsize", type=int, 
            help="number of particles to read per rank in a bunch")
    
    def read(self, columns, full=False):
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

        bunchsize = self.bunchsize
        if full: bunchsize = -1

        args = dict(posdtype=self.posdtype,
             veldtype=self.veldtype,
             massdtype=self.massdtype,
             iddtype=self.iddtype)

        if self.comm.rank == 0:
            datastorage = files.DataStorage(self.path,
                    files.GadgetGroupTabFile, args)
        else:
            datastorage = None
        datastorage = self.comm.bcast(datastorage)

        for round, P in enumerate(
                datastorage.iter(comm=self.comm, 
                    columns=newcolumns, bunchsize=bunchsize)):
            P = dict(zip(newcolumns, P))
            if 'Position' in P:
                P['Position'] /= self.mpch
            if 'Velocity' in P:
                raise KeyError('Velocity is not yet supported')

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= self.BoxSize[dir]

            yield [P[key] for key in columns]
