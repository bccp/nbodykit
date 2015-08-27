from nbodykit.plugins import InputPainter, BoxSizeParser
from nbodykit import files
from itertools import izip
import numpy
import logging

#------------------------------------------------------------------------------          
class CollapsedHaloPainter(InputPainter):
    field_type = "CollapsedHaloMatter"
    
    @classmethod
    def register(kls):
        h = kls.add_parser(kls.field_type)
        
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
        h.set_defaults(klass=kls)
    
    def paint(self, pm):
        if pm.comm.rank == 0:
            hf = files.HaloFile(self.pathhalo)
            nhalo = hf.nhalo
            halopos = numpy.float32(hf.read_pos())
            halomass = numpy.float32(hf.read_mass() * self.m0)
            logmass = numpy.log10(halomass)
            halomask = logmass > self.logMmin
            halomask &= logmass < self.logMmax
            logging.info("total number of halos in mass range is %d" % halomask.sum())
        else:
            halopos = None
            halomask = None

        halopos = pm.comm.bcast(halopos)
        halomask = pm.comm.bcast(halomask)

        Ntot = 0
        for round, (P, PL) in enumerate(izip(
                    files.read(pm.comm, self.pathmatter, files.TPMSnapshotFile, 
                        columns=['Position', 'Velocity'], bunchsize=ns.bunchsize),
                    files.read(pm.comm, self.pathlabel, files.HaloLabelFile, 
                        columns=['Label'], bunchsize=ns.bunchsize),
                    )):
            mask = PL['Label'] != 0
            mask &= halomask[PL['Label']]
            logging.info("Number of particles in halos is %d" % mask.sum())

            P['Position'][mask] = halopos[PL['Label'][mask]]

            if self.rsd is not None:
                dir = 'xyz'.index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= 1.0 # enforce periodic boundary conditions

            P['Position'] *= self.BoxSize
            layout = pm.decompose(P['Position'])
            tpos = layout.exchange(P['Position'])
            #print tpos.shape
            pm.paint(tpos)
            npaint = pm.comm.allreduce(len(tpos)) 
            nread = pm.comm.allreduce(len(P['Position'])) 
            if pm.comm.rank == 0:
                logging.info('round %d, npaint %d, nread %d' % (round, npaint, nread))
            Ntot = Ntot + nread

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot

