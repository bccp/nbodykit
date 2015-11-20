from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser
from nbodykit import files
from itertools import izip
import numpy
import logging

logger = logging.getLogger('CollapsedHalo')
    
class CollapsedHaloDataSource(DataSource):
    field_type = "CollapsedHaloMatter"
    
    @classmethod
    def register(kls):
        h = kls.add_parser()
        
        h.add_argument("pathhalo", help="path to file")
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("logMmin", type=float, help="log10 min mass")
        h.add_argument("logMmax", type=float, help="log10 max mass")
        h.add_argument("m0", type=float, help="mass mass of a particle")
        h.add_argument("pathmatter", help="path to matter file")
        h.add_argument("pathlabel", help="path to label file")
        
        h.add_argument("-rsd", 
            choices="xyz", help="direction to do redshift distortion")
    
    def read(self, columns, comm, bunchsize):
        if comm.rank == 0:
            hf = files.HaloFile(self.pathhalo)
            nhalo = hf.nhalo
            halopos = numpy.float32(hf.read_pos())
            halomass = numpy.float32(hf.read_mass() * self.m0)
            logmass = numpy.log10(halomass)
            halomask = logmass > self.logMmin
            halomask &= logmass < self.logMmax
            logger.info("total number of halos in mass range is %d" % halomask.sum())
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
            logger.info("Number of particles in halos is %d" % mask.sum())

            P['Position'][mask] = halopos[PL['Label'][mask]]

            if self.rsd is not None:
                dir = 'xyz'.index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= 1.0 # enforce periodic boundary conditions

            P['Position'] *= self.BoxSize
            P['Velocity'] *= self.BoxSize

            if comm.rank == 0:
                logger.info('round %d, nread %d' % (round, nread))

            yield [P.get(key, None) for key in columns]


