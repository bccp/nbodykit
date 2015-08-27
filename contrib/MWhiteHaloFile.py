from nbodykit.plugins import InputPainter, BoxSize_t
import numpy
import logging
from nbodykit.utils import selectionlanguage

class MWhiteHaloFile(object):
    """
    Halo catalog file using Martin White's format    
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
        
        h = kls.add_parser(kls.field_type)
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=BoxSize_t,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
            
        h.add_argument("-rsd", 
            choices="xyz", help="direction to do redshift distortion")
        h.add_argument("-select", default=None, type=selectionlanguage.Query,
            help='row selection based on logmass, e.g. logmass > 13 and logmass < 15')
        h.set_defaults(klass=kls)
    
    def read(self, columns, comm):
        dtype = numpy.dtype([
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3)),
            ('logmass', 'f4'),
            ('Mass', 'f4'), ])
            
        if comm.rank == 0:
            hf = MWhiteHaloFile(self.path)
            nhalo = hf.nhalo
            data = numpy.empty(nhalo, dtype)
            
            data['Position']= numpy.float32(hf.read_pos())
            data['Velocity']= numpy.float32(hf.read_vel())
            # unweighted!
            data['Mass'] = 1.0
            data['logmass'] = numpy.log10(numpy.float32(hf.read_mass()))
            
            # select based on selection conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            logging.info("total number of halos in mass range is %d / %d" % (len(data), nhalo))
        else:
            data = numpy.empty(0, dtype=dtype)

        data['Position'][:] *= self.BoxSize
        data['Velocity'][:] *= self.BoxSize

        yield data
