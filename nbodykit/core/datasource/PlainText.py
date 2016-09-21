from nbodykit.core import DataSource
from nbodykit.utils import selectionlanguage
import numpy

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
        
        # positional arguments
        self.path = path
        self.names = names
        self.BoxSize = BoxSize
        
        # keywords
        self.usecols = usecols
        self.poscols = poscols
        self.velcols = velcols
        self.rsd = rsd
        self.posf = posf
        self.velf = velf
        self.select = select
    
    @classmethod
    def fill_schema(cls):
        
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
        from nbodykit.ndarray import extend_dtype
        
        # read in the plain text file as a recarray
        kwargs = {}
        kwargs['comments'] = '#'
        kwargs['names'] = self.names
        kwargs['usecols'] = self.usecols
        data = numpy.recfromtxt(self.path, **kwargs)
        nobj = len(data)
        
        # copy the data
        new_dtypes = [('Position', ('f4', len(self.poscols)))]
        if self.velcols is not None or self.rsd is not None:
            new_dtypes += [('Velocity', ('f4', len(self.velcols)))]
        data = extend_dtype(data, new_dtypes)
           
        # get position and velocity, if we have it
        pos = numpy.vstack(data[k] for k in self.poscols).T.astype('f4')
        pos *= self.posf
        if self.velcols is not None or self.rsd is not None:
            vel = numpy.vstack(data[k] for k in self.velcols).T.astype('f4')
            vel *= self.velf
            data['Velocity'] = vel

        # do RSD
        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= self.BoxSize[dir]
        data['Position'] = pos
        
        # select based on input conditions
        if self.select is not None:
            mask = self.select.get_mask(data)
            data = data[mask]
        self.logger.info("total number of objects selected is %d / %d" % (len(data), nobj))
        
        toret = {}
        for name in data.dtype.names:
            toret[name] = data[name].copy()
        
        return toret
