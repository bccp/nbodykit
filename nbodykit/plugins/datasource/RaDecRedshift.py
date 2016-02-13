from nbodykit.extensionpoints import DataSource
from nbodykit.utils import selectionlanguage

import logging
import numpy

logger = logging.getLogger('RaDecRedshift')
  
def list_str(value):
    return value.split()
    
class RaDecRedshiftDataSource(DataSource):
    """
    DataSource to read in (ra, dec, redshift, weigh) -- only handles 
    the reading of the files
    """
    plugin_name = "RaDecRedshift"
            
    @classmethod
    def register(kls):
        
        p = kls.parser
        
        # required arguments
        p.add_argument("path", 
            help="file path specifying the data to read")
        p.add_argument("names", type=list_str, 
            help="names of columns in text file or name of the data group in hdf5 file")
                
        # optional arguments
        p.add_argument("-usecols", type=list_str, metavar="ra dec z",
            help="only read these columns from file")
        p.add_argument("-sky_cols", type=list_str, default=['ra','dec'], metavar="ra dec",
            help="names of the columns specifying the sky coordinates")
        p.add_argument("-z_col", type=str, default='z',
            help="name of the column specifying the redshift coordinate")
        p.add_argument("-weight_col", type=str, default='z',
            help="name of the column specifying the a weight for each object")
        p.add_argument('-degrees', action='store_true',
            help='if input (ra,dec) are in degrees, set this flag to convert them to radians')
        p.add_argument("-select", default=None, type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
        p.add_argument("-ftype", default='auto', choices=['hdf5', 'text', 'auto'], 
            help='format of the Pandas storage container. auto is to guess from the file name.')
                  
    def readall(self, columns):
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
        if self.ftype == 'hdf5':
            # read in the hdf5 file using pandas
            data = pd.read_hdf(self.path, self.names[0], columns=self.usecols)
        elif self.ftype == 'text':
            # read in the plain text file using pandas
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
        
        # rescale the angles
        if self.degrees:
            data[self.sky_cols] *= numpy.pi/180.
        
        # get the (ra, dec, z) coords
        cols = self.sky_cols + [self.z_col]
        pos = data[cols].values.astype('f4')
        
        # get the weights
        w = numpy.ones(len(pos))
        if self.weight_col is not None:
            w = data[self.weight_col].values.astype('f4')
            
        P = {}
        P['Position'] = pos
        P['Weight'] = w

        return [P[key] for key in columns]