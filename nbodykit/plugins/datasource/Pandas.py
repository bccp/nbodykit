from nbodykit.extensionpoints import DataSource
import numpy
import logging
from nbodykit.utils import selectionlanguage

logger = logging.getLogger('Pandas')
         
class PandasDataSource(DataSource):
    """
    Class to read data from a plaintext file using
    `pandas.read_csv` or a 'pandas-flavored' `HDF5` file
    using `pandas.read_hdf`
    
    Reading method is guessed from the file type, or
    specified via the `ftype` argument. 

    Notes
    -----
    *   `pandas` must be installed to use
    *   when reading plaintext files, the file is assumed to be space-separated
    *   commented lines must begin with `#`, with all other lines
        providing data values to be read
    *   `names` parameter must be equal to the number of data
        columns, otherwise behavior is undefined
    """
    plugin_name = "Pandas"
    
    def __init__(self, path, names, BoxSize, 
                    usecols=None, poscols=['x','y','z'], velcols=None, 
                    rsd=None, posf=1., velf=1., select=None, ftype='auto'):        
        pass
    
    @classmethod
    def register(cls):
        """
        Fill the attribute schema associated with this class
        """
        s = cls.schema
        s.description = "read data from a plaintext or HDF5 file using Pandas"
        
        s.add_argument("path", type=str,
            help="the file path to load the data from")
        s.add_argument("names", type=str, nargs='+', 
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
        s.add_argument("ftype", choices=['hdf5', 'text', 'auto'], 
            help='format of the Pandas storage container. auto is to guess from the file name.')
    
    def readall(self):
        """
        Read all available data, returning a dictionary
        
        This provides ``Position`` and optionally ``Velocity`` columns
        """
        try:
            import pandas as pd
        except:
            name = self.__class__.__name__
            raise ImportError("pandas must be installed to use %s" %name)
                
        if self.ftype == 'auto':
            if self.path.endswith('.hdf5'):
                self.ftype = 'hdf5'
            else: 
                self.ftype = 'text'
                
        # read in the hdf5 file using pandas
        if self.ftype == 'hdf5':
            data = pd.read_hdf(self.path, self.names[0], columns=self.usecols)
        # read in the plain text file using pandas
        elif self.ftype == 'text':
            kwargs = {}
            kwargs['comment'] = '#'
            kwargs['names'] = self.names
            kwargs['header'] = None
            kwargs['engine'] = 'c'
            kwargs['delim_whitespace'] = True
            kwargs['usecols'] = self.usecols
            data = pd.read_csv(self.path, **kwargs)

        nobj = len(data)
        
        # select based on input conditions
        if self.select is not None:
            mask = self.select.get_mask(data)
            data = data[mask]
        logger.info("total number of objects selected is %d / %d" % (len(data), nobj))

        toret = {}
        
        # get position and velocity, if we have it
        pos = data[self.poscols].values.astype('f4')
        pos *= self.posf
        if self.velcols is not None or self.rsd is not None:
            vel = data[self.velcols].values.astype('f4')
            vel *= self.velf
            toret['Velocity'] = vel

        # shift position by RSD
        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= self.BoxSize[dir]
        toret['Position'] = pos
        
        return toret

