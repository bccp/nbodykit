from nbodykit.plugins import InputPainter, BoxSize_t

import numpy
import logging
from nbodykit import files 
from nbodykit.utils import selectionlanguage
  
class HaloFilePainter(InputPainter):
    field_type = "HaloFile"
    
    @classmethod
    def register(kls):
        
        args    = kls.field_type+":path:m0:BoxSize"
        options = "[:-rsd=[x|y|z]][:-select=conditions][:-massweighted]"
        h       = kls.add_parser(kls.field_type, usage=args+options)
        
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=BoxSize_t,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("m0", type=float, help="mass mass of a particle")
        h.add_argument("-massweighted", action='store_true', default=False, 
            help="weight halos by mass?")
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
        h.add_argument("-select", default=None, type=selectionlanguage.Query,
            help='row selection based on logmass, e.g. logmass > 13 and logmass < 15')
        h.set_defaults(klass=kls)
    
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
            data = numpy.empty(nhalo, dtype)
            
            data['Position']= numpy.float32(hf.read('Position'))
            data['Velocity']= numpy.float32(hf.read('Velocity'))
            data['Mass'] = numpy.float32(hf.read('Mass') * self.m0)
            data['logmass'] = numpy.log10(numpy.float32(hf.read('Mass') * self.m0))
            data['length'] = numpy.float32(hf.read('Mass'))

            # select based on selection conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            logging.info("total number of halos in mass range is %d / %d" % (len(data), nhalo))
        else:
            data = numpy.empty(0, dtype=dtype)

        if not self.massweighted:
            data['Mass'] = 1.0

        # put position into units of BoxSize before gridding
        data['Position'] *= self.BoxSize
        # put velocity into units of BoxSize before gridding
        data['Velocity'] *= self.BoxSize

        yield data
        
