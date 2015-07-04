from nbodykit.plugins import InputPainter
import numpy
import logging

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
            usage=kls.field_type+":path:logMmin:logMmax[:-rsd=[x|y|z]]",
            )
        h.add_argument("path", help="path to file")
        h.add_argument("logMmin", type=float, help="log10 min mass")
        h.add_argument("logMmax", type=float, help="log10 max mass")
        h.add_argument("-rsd", 
            choices="xyz", help="direction to do redshift distortion")
        h.set_defaults(klass=kls)
    
    def paint(self, ns, pm):
        if pm.comm.rank == 0:
            hf = MWhiteHaloFile(self.path)
            nhalo = hf.nhalo
            halopos = numpy.float32(hf.read_pos())
            halovel = numpy.float32(hf.read_vel())
            halomass = numpy.float32(hf.read_mass())
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
            halopos[:, dir] %= 1.0 # enforce periodic boundary conditions
        halopos *= ns.BoxSize

        layout = pm.decompose(halopos)
        tpos = layout.exchange(halopos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot