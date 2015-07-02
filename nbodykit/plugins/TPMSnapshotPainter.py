from nbodykit.plugins import InputPainter

# These shall be merged here.
from nbodykit import files 
import logging

#------------------------------------------------------------------------------
class TPMSnapshotPainter(InputPainter):
    field_type = "TPMSnapshot"
    
    @classmethod
    def register(kls):
        h = kls.add_parser(kls.field_type, 
            usage=kls.field_type+":path[:-rsd=[x|y|z]]")
        h.add_argument("path", help="path to file")
        h.add_argument("-rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
        h.set_defaults(klass=kls)

    def paint(self, ns, pm):
        pm.real[:] = 0
        Ntot = 0
        for round, P in enumerate(
                files.read(pm.comm, 
                    self.path, 
                    files.TPMSnapshotFile, 
                    columns=['Position', 'Velocity'], 
                    bunchsize=ns.bunchsize)):

            nread = pm.comm.allreduce(len(P['Position'])) 

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]

            P['Position'] *= ns.BoxSize
            layout = pm.decompose(P['Position'])
            tpos = layout.exchange(P['Position'])
            #print tpos.shape
            pm.paint(tpos)
            npaint = pm.comm.allreduce(len(tpos)) 
            if pm.comm.rank == 0:
                logging.info('round %d, npaint %d, nread %d' % (round, npaint, nread))
            Ntot = Ntot + nread
        return Ntot

#------------------------------------------------------------------------------
