import numpy
from nbodykit.plugins import InputPainter, BoxSizeParser
from nbodykit.utils import selectionlanguage
import os.path
import logging

class PainterPlugin(InputPainter):
    field_type = "MBIIDMO"
    
    @classmethod
    def register(kls):
        h = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("simulation", help="name of simulation", choices=["dmo", "mb2"])
        h.add_argument("type", help="type of objects", choices=["Centrals", "Satellites", "Both"])
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
            
        h.add_argument("-rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
        h.add_argument("-posf", default=0.001, 
                help="factor to scale the positions", type=float)
        h.add_argument("-velf", default=0.001, 
                help="factor to scale the velocities", type=float)
        h.add_argument("-select", default=None, type=selectionlanguage.Query,
            help="row selection based on logmass or subtype, e.g. logmass > 13 and logmass < 15 and subtype == 'A'")

    def read_block(self, block, dtype, optional=False):
            
        if self.type == "Both":
            types = "Centrals", "Satellites"
        else:
            types = [self.type]
            
        try:
            return numpy.concatenate([
                numpy.fromfile(self.path + '/' + type + '/' + self.simulation + '_' + block, dtype=dtype)
                for type in types], axis=0)
        except Exception as e:
            if not optional:
                raise RuntimeError("error reading %s: %s" %(block, str(e)))
            else:
                return numpy.nan
            
            
    def paint(self, pm):
        dtype = numpy.dtype([
                ('position', ('f4', 3)),
                ('velocity', ('f4', 3)),
                ('logmass', 'f8'), 
                ('subtype', 'S1'),
                ('magnitude', ('f8', 5))])
                
        if pm.comm.rank == 0:
            
            pos = self.read_block('pos', dtype['position'])
            data = numpy.empty(len(pos), dtype=dtype)
            data['position'] = pos
            data['velocity'] = self.read_block('vel', dtype['velocity'])
            data['logmass'] = numpy.log10(self.read_block('mass', dtype['logmass']))
            data['subtype'] = self.read_block('subtype', dtype['subtype'], optional=True)
            data['magnitude'] = self.read_block('magnitude', dtype['magnitude'], optional=True)
            
            # select based on selection conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            logging.info("total number of galaxies selected is %d / %d" % (len(data), len(pos)))

            data['position'] *= self.posf
            data['velocity'] *= self.velf
        else:
            data = numpy.empty(0, dtype=dtype)

        Ntot = len(data)
        Ntot = pm.comm.bcast(Ntot)

        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            data['position'][:, dir] += data['velocity'][:, dir]
            data['position'][:, dir] %= self.BoxSize[dir] # enforce periodic boundary conditions

        layout = pm.decompose(data['position'])
        tpos = layout.exchange(data['position'])
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot
