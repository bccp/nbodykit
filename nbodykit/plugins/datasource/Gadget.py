from nbodykit.extensionpoints import DataSource
from nbodykit import files 
import numpy

class GadgetDataSource(DataSource):
    plugin_name = "Gadget"
    
    @classmethod
    def register(kls):
        
        h = kls.parser
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=kls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-ptype", default=[], action='append', type=int,
                help="particle types to use")
        h.add_argument("-posdtype", default='f4', help="dtype of position")
        h.add_argument("-veldtype", default='f4', help="dtype of velocity")
        h.add_argument("-iddtype", default='u8', help="dtype of id")
        h.add_argument("-massdtype", default='f4', help="dtype of mass")
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

        bunchsize = self.bunchsize
        if full: bunchsize = -1
        if full and len(self.ptype) > 1:
            raise ValueError("cannot read multple ptype in a full load")
        stats['Ntot'] = 0
        for ptype in self.ptype:
            args = dict(ptype=ptype,
                 posdtype=self.posdtype,
                 veldtype=self.veldtype,
                 massdtype=self.massdtype,
                 iddtype=self.iddtype)

            if comm.rank == 0:
                datastorage = files.DataStorage(self.path,
                        files.GadgetSnapshotFile, args)
                f0 = files.GadgetSnapshotFile(self.path, 0, args)
                boxsize = f0.header['boxsize']
            else:
                datastorage = None
                boxsize = None
            boxsize = comm.bcast(boxsize)
            datastorage = comm.bcast(datastorage)

            for round, P in enumerate(
                    datastorage.iter(stats=stats, comm=comm, 
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

                yield [P[key] for key in columns]

#------------------------------------------------------------------------------

class GadgetGroupTabDataSource(DataSource):
    plugin_name = "GadgetGroupTab"
    
    @classmethod
    def register(kls):
        
        h = kls.parser
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=kls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-mpch", type=float, default=1000.,
            help="Mpc/h in code unit")
        h.add_argument("-posdtype", default='f4', help="dtype of position")
        h.add_argument("-veldtype", default='f4', help="dtype of velocity")
        h.add_argument("-massdtype", default='f4', help="dtype of mass")
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

        bunchsize = self.bunchsize
        if full: bunchsize = -1

        stats['Ntot'] = 0
        args = dict(posdtype=self.posdtype,
             veldtype=self.veldtype,
             massdtype=self.massdtype,
             iddtype=self.iddtype)

        if comm.rank == 0:
            datastorage = files.DataStorage(self.path,
                    files.GadgetGroupTabFile, args)
        else:
            datastorage = None
        datastorage = comm.bcast(datastorage)

        for round, P in enumerate(
                datastorage.iter(stats=stats, comm=comm, 
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
