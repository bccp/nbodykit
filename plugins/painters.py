from . import PluginMount
from argparse import ArgumentParser

# FIXME: these apply to the individual painters, 
# maybe move to each class?
import numpy
import logging
from nbodykit import files 
from mpi4py import MPI

#------------------------------------------------------------------------------
class InputFieldType:
    """
    Mount point for plugins which refer to the reading of input files 
    and the subsequent painting of those fields.

    Plugins implementing this reference should provide the following 
    attributes:

    field_type : str
        class attribute giving the name of the subparser which 
        defines the necessary command line arguments for the plugin
    
    register : classmethod
        A class method taking no arguments that adds a subparser
        and the necessary command line arguments for the plugin
    
    paint : method
        A method that performs the painting of the field. It 
        takes the following arguments:
            ns : argparse.Namespace
            pm : pypm.particlemesh.ParticleMesh
    """
    __metaclass__ = PluginMount
    
    parser = ArgumentParser("", prefix_chars="-&", add_help=False)
    subparsers = parser.add_subparsers()
    field_type = None

    def __init__(self, string):
        self.string = string
        words = string.split(':')
        
        ns = self.parser.parse_args(words)
        self.painter = ns.klass(ns)
        # steal the paint method
        self.paint = self.painter.paint

    def __eq__(self, other):
        return self.string == other.string

    def __ne__(self, other):
        return self.string != other.string
    
    @classmethod
    def add_parser(kls, name, usage):
        return kls.subparsers.add_parser(name, 
                usage=usage, add_help=False, prefix_chars="&")
    
    @classmethod
    def format_help(kls):
        
        rt = []
        for plugin in kls.plugins:
            k = plugin.field_type
            rt.append(kls.subparsers.choices[k].format_help())

        if not len(rt):
            return "No available input field types"
        else:
            return '\n'.join(rt)
 
#------------------------------------------------------------------------------          
class HaloFilePainter(InputFieldType):
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

        npaint = pm.comm.allreduce(len(tpos), op=MPI.SUM) 
        return Ntot

#------------------------------------------------------------------------------
class TPMSnapshotPainter(InputFieldType):
    field_type = "TPMSnapshot"
    
    def __init__(self, data):
        self.__dict__.update(data.__dict__)
    
    @classmethod
    def register(kls):
        h = kls.add_parser(kls.field_type, 
            usage=kls.field_type+":path[:&rsd=[x|y|z]]")
        h.add_argument("path", help="path to file")
        h.add_argument("&rsd", 
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

            nread = pm.comm.allreduce(len(P['Position']), op=MPI.SUM) 

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]

            P['Position'] *= ns.BoxSize
            layout = pm.decompose(P['Position'])
            tpos = layout.exchange(P['Position'])
            #print tpos.shape
            pm.paint(tpos)
            npaint = pm.comm.allreduce(len(tpos), op=MPI.SUM) 
            if pm.comm.rank == 0:
                logging.info('round %d, npaint %d, nread %d' % (round, npaint, nread))
            Ntot = Ntot + nread
        return Ntot

#------------------------------------------------------------------------------
