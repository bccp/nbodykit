from nbodykit.extensionpoints import DataSource
import numpy
import logging

logger = logging.getLogger('UniformBox')
         
class UniformBoxDataSource(DataSource):
    """
    DataSource to return the following columns:
    
    `Position` : uniformly distributed in box
    `Velocity` : uniformly distributed between ``+/- max_speed``
    `LogMass`  : normally distributed with mean `mu_logM` and std dev `sigma_logM`
    `Mass`     : values corresponding to 10**`LogMass`
    """
    plugin_name = "UniformBox"
    
    def __init__(self, N, BoxSize, max_speed=500., mu_logM=13.5, sigma_logM=0.5, seed=None):        
        
        # initalize a random state for each rank
        if self.seed is not None:
            seed = self.seed*(self.comm.rank+1)
            self.random = numpy.random.RandomState(seed)
        else:
            self.random = numpy.random
            
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "data particles with uniform positions and velocities"
        
        s.add_argument("N", type=int,
            help='the total number of objects in the box')
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("max_speed", type=float, 
            help="use uniform velocities between [-max_speed, max_speed] (in km/s)")
        s.add_argument("mu_logM", type=float, 
            help="the mean log10 mass to use when generating mass values")
        s.add_argument("sigma_logM", type=float, 
            help="the standard deviation of the log10 mass to use when generating mass values")
        s.add_argument("seed", type=int,
            help='the number used to seed the random number generator')
    
    def readall(self):
        """
        Valid columns are:
            `Position` : uniformly distributed in box
            `Velocity` : uniformly distributed between ``+/- max_speed``
            `LogMass`  : normally distributed with mean `mu_logM` and std dev `sigma_logM`
            `Mass`     : values corresponding to 10**`LogMass` 
        """            
        toret = {}
        shape = (self.N, 3)
        
        toret['Position'] = self.random.uniform(size=shape) * self.BoxSize
        toret['Velocity'] = 2*self.max_speed * self.random.uniform(size=shape) - self.max_speed
        toret['LogMass']  = self.random.normal(loc=self.mu_logM, scale=self.sigma_logM, size=self.N)
        toret['Mass']     = 10**(toret['LogMass'])
        
        return toret

