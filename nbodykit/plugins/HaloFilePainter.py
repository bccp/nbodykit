from nbodykit.plugins import InputPainter
from nbodykit.utils.pluginargparse import BoxSizeParser

import numpy
import logging
from nbodykit import files 
from nbodykit.utils import selectionlanguage
  
class HaloFilePainter(InputPainter):
    field_type = "HaloFile"
    
    @classmethod
    def register(kls):
        
        h       = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("m0", type=float, help="mass mass of a particle")
        h.add_argument("-massweighted", action='store_true', default=False, 
            help="weight halos by mass?")
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
        h.add_argument("-select", default=None, type=selectionlanguage.Query,
            help='row selection based on logmass, e.g. logmass > 13 and logmass < 15')
    
    def read(self, columns, comm):
        dtype = numpy.dtype([
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3)),
            ('Mass', ('f4', 3)),
            ('length', 'f4'),
            ('logmass', 'f4')])
        
        if comm.rank == 0:
            hf = files.HaloFile(self.path)
            nhalo = hf.nhalo
            P = numpy.empty(nhalo, dtype)
            
            P['Position']= numpy.float32(hf.read('Position'))
            P['Velocity']= numpy.float32(hf.read('Velocity'))
            P['Mass'] = numpy.float32(hf.read('Mass') * self.m0)
            P['logmass'] = numpy.log10(numpy.float32(hf.read('Mass') * self.m0))
            P['length'] = numpy.float32(hf.read('Mass'))

            # select based on selection conditions
            if self.select is not None:
                mask = self.select.get_mask(P)
                P = P[mask]
            logging.info("total number of halos in mass range is %d / %d" % (len(P), nhalo))
        else:
            P = numpy.empty(0, dtype=dtype)

        if not self.massweighted:
            P['Mass'] = 1.0

        # put position into units of BoxSize before gridding
        P['Position'] *= self.BoxSize
        # put velocity into units of BoxSize before gridding
        P['Velocity'] *= self.BoxSize

        
        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            P['Position'][:, dir] += P['Velocity'][:, dir]
            P['Position'][:, dir] %= self.BoxSize[dir]

        yield P
        
