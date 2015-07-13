import numpy
from nbodykit.plugins import InputPainter
import os.path

class PainterPlugin(InputPainter):
    field_type = "MBIIDMO"
    
    @classmethod
    def register(kls):
        h = kls.add_parser(kls.field_type, 
            usage=kls.field_type+":path:type:logMmin:logMmax:[:-rsd=[x|y|z]][:-posf=0.001][:-velf=0.001]")
        h.add_argument("path", help="path to file")
        h.add_argument("simulation", help="name of simulation", choices=["dmo", "mb2"])
        h.add_argument("type", help="type of objects", choices=["Centrals", "Satellites", "Both"])
        h.add_argument("logMmin", help="log 10 Mmin", type=float)
        h.add_argument("logMmax", help="log 10 Mmax", type=float)
        h.add_argument("-rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
        h.add_argument("-posf", default=0.001, 
                help="factor to scale the positions", type=float)
        h.add_argument("-velf", default=0.001, 
                help="factor to scale the velocities", type=float)
        h.set_defaults(klass=kls)

    def read_block(self, block, dtype):
        if self.type == "Both":
            types = "Centrals", "Satellites"
        else:
            types = [self.type]
        return numpy.concatenate([
            numpy.fromfile(self.path + '/' + type + '/' + self.simulation + '_' + block, dtype=dtype)
            for type in types], axis=0)
            
    def paint(self, ns, pm):
        if pm.comm.rank == 0:
            pos = self.read_block('pos', ('f4', 3))
            vel = self.read_block('vel', ('f4', 3))
            mass = self.read_block('mass', ('f8'))
            logmass = numpy.log10(mass)
            sel = (self.logMmin < logmass) & (self.logMmax > logmass)

            pos = pos[sel]
            vel = vel[sel]
            pos *= self.posf
            vel *= self.velf
        else:
            pos = numpy.empty(0, dtype=('f4', 3))
            vel = numpy.empty(0, dtype=('f4', 3))

        Ntot = len(pos)
        Ntot = pm.comm.bcast(Ntot)

        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            # infer the boxsize from round max value (eh, maybe just use ns.BoxSize?)
            boxsize = numpy.round(numpy.amax(pos[:,dir]))
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= boxsize # enforce periodic boundary conditions

        layout = pm.decompose(pos)
        tpos = layout.exchange(pos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot
