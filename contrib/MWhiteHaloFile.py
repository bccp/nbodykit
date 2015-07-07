from nbodykit.plugins import InputPainter
import numpy
import logging
from nbodykit.utils import selectionlanguage

class MWhiteHaloFile(object):
    """
    Halo catalog file using Martin White's format

    Attributes
    ----------
    nhalo : int
        Number of halos in the file
    
    """
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'r') as ff:
            self._nhalo = int(numpy.fromfile(ff, 'i4', 1)[0])
            
            # first entry of each property is zero 
            # for some reason
            self.nhalo = self._nhalo - 1 
    
    def read(self, column):
        """
        Read a data column from the catalogue

        Parameters
        ----------
        column : string
            column to read: CenterOfMass or Mass
        
        Returns
        -------
            the data column; all halos are returned.

        """
        if column == 'Position':
            return self.read_pos()
        elif column == 'Mass':
            return self.read_mass()
        elif column == 'Velocity':
            return self.read_vel()
        else:
            raise KeyError("column `%s' unknown" % str(column))

    def read_mass(self):
        with open(self.filename, 'r') as ff:
            ff.seek(4, 0)
            return numpy.fromfile(ff, count=self._nhalo, dtype='f4')[1:]

    def read_pos(self):
        with open(self.filename, 'r') as ff:
            ff.seek(4 + 5*self._nhalo * 4, 0)
            return numpy.fromfile(ff, count=self._nhalo, dtype=('f4', 3))[1:]

    def read_vel(self):
        with open(self.filename, 'r') as ff:
            ff.seek(4 + 10*self._nhalo * 4, 0)
            return numpy.fromfile(ff, count=self._nhalo, dtype=('f4', 3))[1:]
            
            
#------------------------------------------------------------------------------          
class MWhiteHaloFilePainter(InputPainter):
    field_type = "MWhiteHaloFile"
    
    @classmethod
    def register(kls):
        
        h = kls.add_parser(kls.field_type, 
            usage=kls.field_type+":path[:-rsd=[x|y|z]][:-select=conditions]",
            )
        h.add_argument("path", help="path to file")
        h.add_argument("-rsd", 
            choices="xyz", help="direction to do redshift distortion")
        h.add_argument("-select", default=None, type=selectionlanguage.Query,
            help='row selection based on logmass, e.g. logmass > 13 and logmass < 15')
        h.set_defaults(klass=kls)
    
    def paint(self, ns, pm):
        dtype = numpy.dtype([
            ('position', ('f4', 3)),
            ('velocity', ('f4', 3)),
            ('logmass', 'f4')])
            
        if pm.comm.rank == 0:
            hf = MWhiteHaloFile(self.path)
            nhalo = hf.nhalo
            data = numpy.empty(nhalo, dtype)
            
            data['position']= numpy.float32(hf.read_pos())
            data['velocity']= numpy.float32(hf.read_vel())
            data['logmass'] = numpy.log10(numpy.float32(hf.read_mass()))
            
            # select based on selection conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            logging.info("total number of halos in mass range is %d / %d" % (len(data), nhalo))
        else:
            data = numpy.empty(0, dtype=dtype)

        Ntot = len(data)
        Ntot = pm.comm.bcast(Ntot)

        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            data['position'][:, dir] += data['velocity'][:, dir]
            data['position'][:, dir] %= 1.0 # enforce periodic boundary conditions
        data['position'] *= ns.BoxSize
        
        layout = pm.decompose(data['position'])
        tpos = layout.exchange(data['position'])
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot