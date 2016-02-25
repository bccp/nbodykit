from nbodykit.extensionpoints import Algorithm, DataSource
from nbodykit import fof
import logging
import numpy

logger = logging.getLogger('FiberCollisions')

def RaDecDataSource(d):
    source = DataSource.registry.RaDecRedshift
    d['unit_sphere'] = True
    return source.from_config(d)
            
class FiberCollisionsAlgorithm(Algorithm):
    """
    Run an angular FOF algorithm to determine fiber collision
    groups from an input catalog, and then assign fibers such that
    the maximum amount of object receive a fiber. This amounts
    to determining the following population of objects:
    
        * population 1: 
            the maximal "clean" sample of objects in which each object is not 
            angularly collided with any other object in this subsample
        * population 2:
            the potentially-collided objects; these objects are those
            that are fiber collided + those that have been "resolved"
            due to multiple coverage in tile overlap regions
    
    See Guo et al. 2010 (http://arxiv.org/abs/1111.6598)for further details
    """
    plugin_name = "FiberCollisions"
    
    def __init__(self, datasource, collision_radius=62/60./60.):
        
        # store collision radius in radians
        self._collision_radius_rad = self.collision_radius * numpy.pi/180.
            
    @classmethod
    def register(cls):

        s = cls.schema
        s.description = "the application of fiber collisions to a galaxy survey"
        
        s.add_argument("datasource", type=RaDecDataSource,
            help='`RaDecRedshift DataSource; run `nbkit.py --list-datasources RaDecRedshift` for details')
        s.add_argument("collision_radius", type=float, 
            help="the size of the angular collision radius (in degrees)")
        
    def run(self):
        """
        Compute the FOF collision groups
        
        Returns
        -------
        labels : array_like, int
            an array of integers specifying the group labels for
            each object in the input DataSource; label == 0 objects
            are no in a group
        collided : array_like, int
            a flag array specifying which objects are collided, i.e.,
            do not receive a fiber
        neighbors : array_like, int
            for those objects that are collided, this gives the index
            of the nearest neighbor on the sky (0-indexed), else it is set to -1
        """
        from nbodykit import fof
        
        # run the angular FoF algorithm to get group labels
        labels = fof.fof(self.datasource, self._collision_radius_rad, 10, comm=self.comm)
        
        print numpy.bincount(labels), len(labels)
        
        # assign the fibers
        if self.comm.rank == 0:
            logger.info("assigning fibers...")
        collided, neighbors = self._assign_fibers(labels)
        f = collided.sum()*1./len(collided)
        
        # print out some info
        if self.comm.rank == 0:
            logger.info("population 1 (clean) size = %d" %(collided^1).sum())
            logger.info("population 2 (collided) size = %d" %collided.sum())
            logger.info("collision fraction = %.4f" %f)
        
        return labels, collided, neighbors

    
    def _assign_multiplets(self, Position, group_indices):
        """
        Internal function to assign the maximal amount of fibers 
        in collision groups of size N > 2
        """
        from scipy.spatial.distance import pdist, squareform
        
        def count(slice, n):
            return n[numpy.nonzero(slice)[0]].sum()
        
        # first shuffle the member ids, so we select random element when tied
        group_ids = list(group_indices)
    
        collided_ids = []
        while len(group_ids) > 1:
       
            # compute dists and find where dists < collision radius
            dists = squareform(pdist(Position[group_ids], metric='euclidean'))
            collisions = (dists > 0.)&(dists <= self._collision_radius_rad)
            
            # total # of collisions for each group member
            n_collisions = numpy.sum(collisions, axis=0)
            
            # total # of collisions for those objects that collide with each group member
            n_other = numpy.apply_along_axis(count, 0, collisions, n_collisions)
            
            # remove object that has most # of collisions 
            # and those colliding objects have least # of collisions
            idx = numpy.where(n_collisions == n_collisions.max())[0]
            ii = numpy.random.choice(numpy.where(n_other[idx] == n_other[idx].min())[0])
            collided_index = idx[ii]  
    
            # make the collided galaxy and remove from group
            collided_id = group_ids.pop(collided_index)
        
            # only make this a collided object if its n_collisions > 0
            # if n_collisions = 0, then the object can get a fiber for free
            if n_collisions[collided_index] > 0:
                collided_ids.append(collided_id)

        # compute the nearest neighbors
        neighbor_ids = []
        dists = squareform(pdist(Position[group_indices], metric='euclidean'))
        uncollided = [i for i, x in enumerate(group_indices) if x not in collided_ids]
        for idx in collided_ids:
            i = list(group_indices).index(idx)
            neighbor = group_indices[uncollided[dists[i][uncollided].argmin()]]
            neighbor_ids.append(neighbor)
            
        return collided_ids, neighbor_ids
                
    def _assign_pairs(self, groups, N):
        """
        Assign fibers to collision pairs
        """
        # randomly select first/second object of pair
        which = numpy.random.choice([0,1], size=(N==2).sum())
        
        # group numbers of all pairs
        pair_numbers = numpy.nonzero(N==2)[0]

        # randomly select one element of the pair
        collided = []; neighbors = []
        for i in range(self.comm.rank, len(pair_numbers), self.comm.size):
            idx = groups[pair_numbers[i]]
            collided.append(idx[which[i]]) # the collided id
            neighbors.append(idx[which[i]^1]) # the neighbor id is the other object

        return collided, neighbors
        
    def _assign_fibers(self, labels):
        """
        Assign fibers
        """
        import pandas as pd
                
        # group by label to get array indices for each label value
        df = pd.DataFrame(labels, columns=['Label'])
        groupby = df.groupby('Label')
        N = groupby.size().values
        groups = groupby.groups
        
        # setup
        collided = numpy.zeros_like(labels)
        neighbors = numpy.zeros_like(labels)
        
        # assign fibers to the pairs
        index, neighbor_ids = self._assign_pairs(groups, N)
        collided[index] = 1
        neighbors[index] = neighbor_ids
        
        # read in the position data for N > 2
        stats = {}
        [[Position]] = self.datasource.read(['Position'], stats, full=True)
        
        # assign fibers for  N > 2
        for size in numpy.unique(N[N>2]):
            
            group_numbers = numpy.nonzero(N == size)[0]
            for i in range(self.comm.rank, len(group_numbers), self.comm.size):
                group_ids = groups[group_numbers[i]]
                index, neighbor_ids = self._assign_multiplets(Position, group_ids)
                collided[index] = 1
                neighbors[index] = neighbor_ids                
        
        # sum from all ranks
        collided = self.comm.allreduce(collided)
        neighbors = self.comm.allreduce(neighbors)
        neighbors[collided==0] = -1
        
        return collided, neighbors
                

    def save(self, output, data):
        """
        Save the `Label`, `Collided`, and `NeighborID` arrays
        to a pandas HDF file, with key `FiberCollisonGroups`
        """
        import pandas as pd
        
        if self.comm.rank == 0:
            columns = ['Label', 'Collided', 'NeighborID']
            data = dict(zip(columns, data))
            dtype = [(col, data[col].dtype) for col in columns]
            df = pd.DataFrame(data)
            df.to_hdf(output, 'FiberCollisionGroups')

