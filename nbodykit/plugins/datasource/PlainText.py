from nbodykit.extensionpoints import DataSource
from nbodykit.utils import selectionlanguage
import numpy
import logging

logger = logging.getLogger('PlainText')

class PlainTextDataSource(DataSource):
    """
    DataSource to read a plaintext file, using `numpy.recfromtxt`
    to do the reading
    
    Notes
    -----
    *   data file is assumed to be space-separated
    *   commented lines must begin with `#`, with all other lines
        providing data values to be read
    *   `names` parameter must be equal to the number of data
        columns, otherwise behavior is undefined
    """
    plugin_name = "PlainText"
    
    def __init__(self, path, names, BoxSize, 
                    usecols=None, poscols=['x','y','z'], velcols=None, 
                    rsd=None, posf=1., velf=1., select=None):     
        pass
    
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "read data from a plaintext file using numpy"
        
        s.add_argument("path", type=str,
            help="the file path to load the data from")
        s.add_argument("names", type=str, nargs='*', 
            help="names of columns in text file or name of the data group in hdf5 file")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("usecols", type=str, nargs='*',
            help="only read these columns from file")
        s.add_argument("poscols", type=str, nargs=3,
            help="names of the position columns")
        s.add_argument("velcols", type=str, nargs=3,
            help="names of the velocity columns")
        s.add_argument("rsd", type=str, choices="xyz", 
            help="direction to do redshift distortion")
        s.add_argument("posf", type=float, 
            help="factor to scale the positions")
        s.add_argument("velf", type=float, 
            help="factor to scale the velocities")
        s.add_argument("select", type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string, i.e., "Mass > 1e14"')
        
    def readall(self):
        """
        Read all available data, returning a dictionary
        
        This provides ``Position`` and optionally ``Velocity`` columns
        """
        
        # read in the plain text file as a recarray
        kwargs = {}
        kwargs['comments'] = '#'
        kwargs['names'] = self.names
        kwargs['usecols'] = self.usecols
        data = numpy.recfromtxt(self.path, **kwargs)
        nobj = len(data)
        
        # select based on input conditions
        if self.select is not None:
            mask = self.select.get_mask(data)
            data = data[mask]
        logger.info("total number of objects selected is %d / %d" % (len(data), nobj))
        
        toret = {}
        
        # get position and velocity, if we have it
        pos = numpy.vstack(data[k] for k in self.poscols).T.astype('f4')
        pos *= self.posf
        if self.velcols is not None:
            vel = numpy.vstack(data[k] for k in self.velcols).T.astype('f4')
            vel *= self.velf
            toret['Velocity'] = Velocity

        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= self.BoxSize[dir]
        toret['Position'] = pos
        
        return toret

