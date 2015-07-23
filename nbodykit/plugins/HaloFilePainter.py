from nbodykit.plugins import InputPainter

import numpy
import logging
from nbodykit import files 
from nbodykit.utils import selectionlanguage

#------------------------------------------------------------------------------          
class HaloFilePainter(InputPainter):
    field_type = "HaloFile"
    
    @classmethod
    def register(kls):
        
        h = kls.add_parser(kls.field_type, 
            usage=kls.field_type+":path:m0[:-rsd=[x|y|z]][:-select=conditions]",
            )
        h.add_argument("path", help="path to file")
        h.add_argument("m0", type=float, help="mass mass of a particle")
        h.add_argument("-massweighted", action='store_true', default=False, help="weight halos by mass?")
        h.add_argument("-rsd", 
            choices="xyz", help="direction to do redshift distortion")
        h.add_argument("-select", default=None, type=selectionlanguage.Query,
            help='row selection based on logmass, e.g. logmass > 13 and logmass < 15')
        h.set_defaults(klass=kls)
    
    def paint(self, ns, pm):
        dtype = numpy.dtype([
            ('position', ('f4', 3)),
            ('velocity', ('f4', 3)),
            ('length', 'f4'),
            ('logmass', 'f4')])
        
        if pm.comm.rank == 0:
            hf = files.HaloFile(self.path)
            nhalo = hf.nhalo
            data = numpy.empty(nhalo, dtype)
            
            data['position']= numpy.float32(hf.read('Position'))
            data['velocity']= numpy.float32(hf.read('Velocity'))
            data['logmass'] = numpy.log10(numpy.float32(hf.read('Mass') * self.m0))
            data['length'] = numpy.float32(hf.read('Mass'))

            # select based on selection conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            logging.info("total number of halos in mass range is %d / %d" % (len(data), nhalo))
        else:
            data = numpy.empty(0, dtype=dtype)

        if self.massweighted:
            Ntot = data['length'].sum(dtype='f8')
        else:
            Ntot = len(data)
        Ntot = pm.comm.bcast(Ntot)

        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            data['position'][:, dir] += data['velocity'][:, dir]
            data['position'][:, dir] %= 1.0 # enforce periodic boundary conditions
        data['position'] *= ns.BoxSize

        layout = pm.decompose(data['position'])
        tpos = layout.exchange(data['position'])
        tvel = layout.exchange(data['velocity'])

        if self.massweighted:
            weight = layout.exchange(data['length'])
        else:
            weight = 1
        pm.paint(tpos, weight)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot

