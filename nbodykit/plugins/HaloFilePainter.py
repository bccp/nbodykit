from nbodykit.plugins import InputPainter

# FIXME: these apply to the individual painters, 
# maybe move to each class?
import numpy
import logging

#------------------------------------------------------------------------------          
class HaloFilePainter(InputPainter):
    field_type = "HaloFile"
    
    def __init__(self, data): 
        self.__dict__.update(data.__dict__)
                
    @classmethod
    def register(kls):
        
        h = kls.add_parser(kls.field_type, 
            usage=kls.field_type+":path:logMmin:logMmax:m0[:&rsd=[x|y|z]]",
            )
        h.add_argument("path", help="path to file")
        h.add_argument("logMmin", type=float, help="log10 min mass")
        h.add_argument("logMmax", type=float, help="log10 max mass")
        h.add_argument("m0", type=float, help="mass mass of a particle")
        h.add_argument("&rsd", 
            choices="xyz", help="direction to do redshift distortion")
        h.set_defaults(klass=kls)
    
    def paint(self, ns, pm):
        if pm.comm.rank == 0:
            hf = files.HaloFile(self.path)
            nhalo = hf.nhalo
            halopos = numpy.float32(hf.read_pos())
            halovel = numpy.float32(hf.read_vel())
            halomass = numpy.float32(hf.read_mass() * self.m0)
            logmass = numpy.log10(halomass)
            mask = logmass > self.logMmin
            mask &= logmass < self.logMmax
            halopos = halopos[mask]
            halovel = halovel[mask]
            logging.info("total number of halos in mass range is %d" % mask.sum())
        else:
            halopos = numpy.empty((0, 3), dtype='f4')
            halovel = numpy.empty((0, 3), dtype='f4')
            halomass = numpy.empty(0, dtype='f4')

        Ntot = len(halopos)
        Ntot = pm.comm.bcast(Ntot)

        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            halopos[:, dir] += halovel[:, dir]
        halopos *= ns.BoxSize

        layout = pm.decompose(halopos)
        tpos = layout.exchange(halopos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot

