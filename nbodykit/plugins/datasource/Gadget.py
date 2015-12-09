from nbodykit.extensionpoints import DataSource
from nbodykit import files 
import numpy

class GadgetDataSource(DataSource):
    field_type = "Gadget"
    
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

        if 'Mass' in newcolumns:
            newcolumns.remove('Mass')
        if 'Weight' in newcolumns:
            newcolumns.remove('Weight')

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
                    print P['Position']
                if 'Velocity' in P:
                    raise KeyError('Velocity is not yet supported')

                P['Mass'] = numpy.ones(stats['Ncurrent'], dtype='u1')
                if self.rsd is not None:
                    dir = "xyz".index(self.rsd)
                    P['Position'][:, dir] += P['Velocity'][:, dir]
                    P['Position'][:, dir] %= self.BoxSize[dir]

                yield [P[key] for key in columns]

#------------------------------------------------------------------------------
