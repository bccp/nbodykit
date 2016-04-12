from nbodykit.extensionpoints import DataSource
from nbodykit.utils import selectionlanguage
import logging
import numpy
import itertools

logger = logging.getLogger('RaDecRedshift')

    
class RaDecRedshiftDataSource(DataSource):
    """
    DataSource designed to handle reading (ra, dec, redshift)
    from a plaintext file, using `pandas.read_csv`
    
    *   Returns the Cartesian coordinates corresponding to 
        (ra, dec, redshift) as the `Position` column.
    *   If `unit_sphere = True`, the Cartesian coordinates
        are on the unit sphere, so the the redshift information
        is not used
    """
    plugin_name = "RaDecRedshift"
    
    def __init__(self, path, names, unit_sphere=False, 
                    usecols=None, sky_cols=['ra','dec'], z_col='z', 
                    weight_col=None, degrees=False, select=None, nbar_col=None):       

        # setup the cosmology
        if not self.unit_sphere:
            if self.cosmo is None:
                raise ValueError("please specify a input Cosmology to use in `RaDecRedshift`")
            
            # sample the cosmology's comoving distance
            self.cosmo.sample('comoving_distance', numpy.logspace(-5, 1, 1024))
        else:
            # unit sphere fits in box of size L = 2
            self.BoxSize = numpy.array([2., 2., 2.])

  
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "read (ra, dec, z) from a plaintext file, returning Cartesian coordinates"
        
        # required
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("names", type=str, nargs='+', help="the names of columns in text file")
        
        # optional
        s.add_argument('unit_sphere', type=bool, 
            help='if True, return Cartesian coordinates on the unit sphere')
        s.add_argument("usecols", type=str, nargs='*', 
            help="only read these columns from file")
        s.add_argument("sky_cols", type=str, nargs='*',
            help="names of the columns specifying the sky coordinates")
        s.add_argument("z_col", type=str,
            help="name of the column specifying the redshift coordinate")
        s.add_argument("weight_col", type=str,
            help="name of the column specifying the `weight` for each object")
        s.add_argument("nbar_col", type=str,
            help="name of the column specifying the `nbar` value for each object")
        s.add_argument('degrees', type=bool,
            help='set this flag if the input (ra, dec) are in degrees')
        s.add_argument("select", type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
                  
    def _to_cartesian(self, coords):
        """
        Convert the (ra, dec, redshift) coordinates to cartesian coordinates,
        scaled to the comoving distance if `unit_sphere = False`, else
        on the unit sphere
        
        Notes
        -----
        Input angles `ra` and `dec` (first 2 columns of `coords`)  
        are assumed to be in radians
        
        Parameters
        -----------
        coords : array_like, (N, 3)
            the input coordinates with the columns giving (ra, dec, redshift),
            where ``ra`` and ``dec`` are in radians
        
        Returns
        -------
        pos : array_like, (N,3)
            the cartesian position coordinates, where columns represent ``x``, 
            ``y``, and ``z``
        """
        ra, dec, redshift = coords.T
        
        x = numpy.cos( dec ) * numpy.cos( ra )
        y = numpy.cos( dec ) * numpy.sin( ra )
        z = numpy.sin( dec )
                
        pos = numpy.vstack([x,y,z])
        if not self.unit_sphere:
            pos *= self.cosmo.comoving_distance(redshift)
        
        return pos.T
        
    def readall(self):
        """
        Read all available data, returning a dictionary
        
        This provides the following columns:
        
            ``Ra``  : right ascension (in radians)
            ``Dec`` : declination (in radians)  
            ``Redshift`` : the redshift
            ``Position`` : cartesian coordinates computed from angular coords + redshift
        
        And optionally, the `Weight` and `Nbar` columns
        """  
        try:
            import pandas as pd
        except:
            name = self.__class__.__name__
            raise ImportError("pandas must be installed to use %s" %name)

        # read in the plain text file using pandas
        kwargs = {}
        kwargs['comment'] = '#'
        kwargs['names'] = self.names
        kwargs['header'] = None
        kwargs['engine'] = 'c'
        kwargs['delim_whitespace'] = True
        kwargs['usecols'] = self.usecols
        
        # iterate through in parallel
        data = pd.read_csv(self.path, **kwargs)

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

        toret             = {}
        toret['Ra']       = pos[:,0]
        toret['Dec']      = pos[:,1]
        toret['Redshift'] = pos[:,2]
        toret['Position'] = self._to_cartesian(pos)

        # optionally, return a weight
        if self.weight_col is not None:
            toret['Weight'] = data[self.weight_col].values.astype('f4')

        # optionally, return nbar
        if self.nbar_col is not None:
            toret['Nbar'] = data[self.nbar_col].values.astype('f4')
    
        return toret