from nbodykit.extensionpoints import DataSource
import numpy
import logging

logger = logging.getLogger('UniformBox')

def get_Nlocal(N, size, rank):
    """
    Return the local number of objects on each rank, given
    the desired total `N`, the communicator `size`, and the
    `rank`
    """
    Neach_section, extras = divmod(N, size)
    section_sizes = extras * [Neach_section+1] + (size-extras) * [Neach_section]
    return section_sizes[rank]
         
class UniformBoxDataSource(DataSource):
    """
    Class to return particles uniformly distributed
    in a box between [0, BoxSize], and velocities 
    uniformly distributed between ``[-max_speed, max_speed]``
    """
    plugin_name = "UniformBox"
    
    def __init__(self, N, BoxSize, max_speed=500., seed=1234):        
        
        # initalize a random state for each rank
        seed = self.seed*(self.comm.rank+1)
        self.random = numpy.random.RandomState(seed)
        
        # local number of particles on each rank
        self.Nlocal = get_Nlocal(self.N, self.comm.size, self.comm.rank)
    
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
        s.add_argument("seed", type=int,
            help='the number used to seed the random number generator')
    
    def readall(self):
        """
        Return `Position` and `Velocity` distributed normally in the box
        """            
        toret = {}
        shape = (self.Nlocal, 3)
        
        toret['Position'] = self.random.uniform(size=shape) * self.BoxSize
        toret['Velocity'] = 2*self.max_speed * self.random.uniform(size=shape) - self.max_speed
        
        return toret

