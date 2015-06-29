import numpy
import logging
from nbodykit import files 
from mpi4py import MPI

class HaloFilePainter(object):
    
    @classmethod
    def register(kls, inputdesc):
        h = inputdesc.add_parser("HaloFile", 
            usage="HaloFile:path:min:max:m0[:&rsd=[x|y|z]]",
            )
        h.add_argument("path", help="path to file")
        h.add_argument("logMmin", type=float, help="log10 min mass")
        h.add_argument("logMmax", type=float, help="log10 max mass")
        h.add_argument("m0", type=float, help="mass mass of a particle")
        h.add_argument("&rsd", 
            choices="xyz", help="direction to do redshift distortion")

        h.set_defaults(painter=kls.paint)
    
    @classmethod
    def paint(kls, ns, desc, pm):
        if pm.comm.rank == 0:
            hf = files.HaloFile(desc.path)
            nhalo = hf.nhalo
            halopos = numpy.float32(hf.read_pos())
            halomass = numpy.float32(hf.read_mass() * desc.m0)
            logmass = numpy.log10(halomass)
            mask = logmass > desc.logMmin
            mask &= logmass < desc.logMmax
            halopos = halopos[mask]
            logging.info("total number of halos in mass range is %d" % mask.sum())
        else:
            halopos = numpy.empty((0, 3), dtype='f4')
            halomass = numpy.empty(0, dtype='f4')

        Ntot = len(halopos)
        Ntot = pm.comm.bcast(Ntot)

        halopos *= ns.BoxSize

        layout = pm.decompose(halopos)
        tpos = layout.exchange(halopos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos), op=MPI.SUM) 
        return Ntot