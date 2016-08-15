from nbodykit.core import DataSource
import numpy
         
class UnitTestDataSource(DataSource):
    """
    A DataSource to be used in unit testing. 
    
    Given a size and list of data types, this returns random 
    numbers using :func:`numpy.random.random` (using the specified seed) 
    for each desired column in the data type. 
    """
    plugin_name = "UnitTestDataSource"
    
    def __init__(self, N, dtype, seed=None):        
        from astropy.utils.misc import NumpyRNGContext
        
        self.N     = N
        self.dtype = dtype
        self.seed  = seed
        
        # generate the data with a fixed seed
        with NumpyRNGContext(self.seed):
            
            # fill the structured array with random numbers
            self.data = numpy.zeros(self.N, dtype=self.dtype)
            for name in self.data.dtype.names:
                self.data[name] = numpy.random.random(size=self.data[name].shape)
        
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "data source for unit testing purposes, returning random numbers"
        
        s.add_argument("N", type=int,
            help='the total size of the datasource')
        s.add_argument('dtype',
            help='the list of data type tuples')
        s.add_argument("seed", type=int,
            help='the number used to seed the random number generator')
    
    def readall(self):
        """
        Return the columns in the structured array
        """ 
        toret = {}
        for name in self.data.dtype.names:
            toret[name] = self.data[name]
            
        return toret

