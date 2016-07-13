from nbodykit.extensionpoints import DataSource
from nbodykit import files
from itertools import izip
import numpy
    
class CollapsedHaloDataSource(DataSource):
    plugin_name = "CollapsedHaloMatter"
    
    def __init__(self, pathhalo, BoxSize, logMmin, logMmax, m0, pathmatter, pathlabel, rsd=None):
        pass
    
    @classmethod
    def register(cls):
        s = cls.schema
        
        s.add_argument("pathhalo", help="path to file")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("logMmin", type=float, help="log10 min mass")
        s.add_argument("logMmax", type=float, help="log10 max mass")
        s.add_argument("m0", type=float, help="mass mass of a particle")
        s.add_argument("pathmatter", help="path to matter file")
        s.add_argument("pathlabel", help="path to label file")
        s.add_argument("rsd", choices="xyz", help="direction to do redshift distortion")
    
    def parallel_read(self, columns, full=False):
        if comm.rank == 0:
            hf = files.HaloFile(self.pathhalo)
            nhalo = hf.nhalo
            halopos = numpy.float32(hf.read_pos())
            halomass = numpy.float32(hf.read_mass() * self.m0)
            logmass = numpy.log10(halomass)
            halomask = logmass > self.logMmin
            halomask &= logmass < self.logMmax
            self.logger.info("total number of halos in mass range is %d" % halomask.sum())
        else:
            halopos = None
            halomask = None

        halopos = comm.bcast(halopos)
        halomask = comm.bcast(halomask)

        for round, (P, PL) in enumerate(izip(
                    files.read(comm, self.pathmatter, files.TPMSnapshotFile, 
                        columns=['Position', 'Velocity'], bunchsize=ns.bunchsize),
                    files.read(comm, self.pathlabel, files.HaloLabelFile, 
                        columns=['Label'], bunchsize=ns.bunchsize),
                    )):
            mask = PL['Label'] != 0
            mask &= halomask[PL['Label']]
            self.logger.info("Number of particles in halos is %d" % mask.sum())

            P['Position'][mask] = halopos[PL['Label'][mask]]

            if self.rsd is not None:
                dir = 'xyz'.index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= 1.0 # enforce periodic boundary conditions

            P['Position'] *= self.BoxSize
            P['Velocity'] *= self.BoxSize

            if comm.rank == 0:
                self.logger.info('round %d, nread %d' % (round, nread))

            yield [P.get(key, None) for key in columns]


