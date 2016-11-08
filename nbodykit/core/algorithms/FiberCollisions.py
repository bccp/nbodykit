from nbodykit.core import Algorithm, DataSource
from nbodykit import fof, utils
import numpy

def RaDecDataSource(d):
    from nbodykit import plugin_manager
    source = plugin_manager.get_plugin('RaDecRedshift')
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
    
    def __init__(self, datasource, collision_radius=62/60./60., seed=None):
        
        # set the input parameters
        self.datasource       = datasource
        self.collision_radius = collision_radius
        self.seed             = seed
        
        # store collision radius in radians
        self._collision_radius_rad = self.collision_radius * numpy.pi/180.
        if self.comm.rank == 0: 
            self.logger.info("collision radius in degrees = %.4f" %collision_radius)
            
        # create the local random seed from the global seed and comm size
        self.local_seed = utils.local_random_seed(self.seed, self.comm)
        self.logger.info("local_seed = %d" %self.local_seed)
        
    @classmethod
    def fill_schema(cls):

        s = cls.schema
        s.description = "the application of fiber collisions to a galaxy survey"
        
        s.add_argument("datasource", type=RaDecDataSource,
            help='`RaDecRedshift DataSource; run `nbkit.py --list-datasources RaDecRedshift` for details')
        s.add_argument("collision_radius", type=float, 
            help="the size of the angular collision radius (in degrees)")
        s.add_argument("seed", type=int,
            help="seed the random number generator explicitly, for reproducibility")
        
    def run(self):
        """
        Compute the FOF collision groups and assign fibers, such that
        the maximum number of objects receive fibers
        
        Returns
        -------
        result: array_like
            a structured array with 3 fields:
                Label : 
                    the group labels for each object in the input 
                    DataSource; label == 0 objects are not in a group
                Collided : 
                    a flag array specifying which objects are 
                    collided, i.e., do not receive a fiber
                NeighborID : 
                    for those objects that are collided, this 
                    gives the (global) index of the nearest neighbor 
                    on the sky (0-indexed), else it is set to -1
        """
        from nbodykit import fof
        from astropy.utils.misc import NumpyRNGContext
        
        # open a persistent cache
        with self.datasource.keep_cache():
            
            # run the angular FoF algorithm to get group labels
            # labels gives the global group ID corresponding to each object in Position 
            # on this rank
            labels = fof.fof(self.datasource, self._collision_radius_rad, 1, comm=self.comm)

            # assign the fibers (in parallel)
            with NumpyRNGContext(self.local_seed):
                collided, neighbors = self._assign_fibers(labels)
    
        # all reduce to get summary statistics
        N_pop1 = self.comm.allreduce((collided^1).sum())
        N_pop2 = self.comm.allreduce((collided).sum())
        f = N_pop2 * 1. / (N_pop1 + N_pop2)

        # print out some info
        if self.comm.rank == 0:

            self.logger.info("population 1 (clean) size = %d" %N_pop1)
            self.logger.info("population 2 (collided) size = %d" %N_pop2)
            self.logger.info("collision fraction = %.4f" %f)

        # return a structured array
        d = list(zip(['Label', 'Collided', 'NeighborID'], [labels, collided, neighbors]))
        dtype = numpy.dtype([(col, x.dtype) for col, x in d])
        result = numpy.empty(len(labels), dtype=dtype)
        for col, x in d: result[col] = x
        return result

    def _assign_fibers(self, Label):
        """
        Initernal function to divide the data by collision group 
        across ranks and assign fibers, such that the minimum
        number of objects are collided out of the survey
        """
        import mpsort
        from mpi4py import MPI
        
        comm = self.comm
        mask = Label != 0
        dtype = numpy.dtype([
                ('Position', ('f4', 3)),  
                ('Label', ('i4')), 
                ('Rank', ('i4')), 
                ('Index', ('i4')),
                ('Collided', ('i4')),
                ('NeighborID', ('i4'))
                ])
        PIG = numpy.empty(mask.sum(), dtype=dtype)
        PIG['Label'] = Label[mask]
        size = len(Label)
        size = comm.allgather(size)
        Ntot = sum(size)
        offset = sum(size[:comm.rank])
        PIG['Index'] = offset + numpy.where(mask == True)[0]
        del Label
        
        with self.datasource.open() as stream:
            [[Position]] = stream.read(['Position'], full=True)
        PIG['Position'] = Position[mask]
        del Position
        Ntot = comm.allreduce(len(mask))
        Nhalo = comm.allreduce(
            PIG['Label'].max() if len(PIG['Label']) > 0 else 0, op=MPI.MAX) + 1

        # now count number of particles per halo
        PIG['Rank'] = PIG['Label'] % comm.size
        cnt = numpy.bincount(PIG['Rank'], minlength=comm.size)
        Nlocal = comm.allreduce(cnt)[comm.rank]

        # sort by rank and then label
        PIG2 = numpy.empty(Nlocal, PIG.dtype)
        mpsort.sort(PIG, orderby='Rank', out=PIG2, comm=self.comm)
        assert (PIG2['Rank'] == comm.rank).all()
        PIG2.sort(order=['Label'])
        
        if self.comm.rank == 0:
            self.logger.info('total number of collision groups = %d', Nhalo-1)
            self.logger.info("Started fiber assignment")

        # loop over unique group ids
        for group_id in numpy.unique(PIG2['Label']):
            start = PIG2['Label'].searchsorted(group_id, side='left')
            end = PIG2['Label'].searchsorted(group_id, side='right')
            N = end-start
            assert(PIG2['Label'][start:end] == group_id).all()
            
            # pairs (random selection)
            if N == 2:
                
                # randomly choose, with fixed local seed
                which = numpy.random.choice([0,1])
                    
                indices = [start+which, start+(which^1)]
                PIG2['Collided'][indices] = [1, 0]
                PIG2['NeighborID'][indices] = [PIG2['Index'][start+(which^1)], -1]
            # multiplets (minimize collidedness)
            elif N > 2:
                collided, nearest = self._assign_multiplets(PIG2['Position'][start:end])
                PIG2['Collided'][start:end] = collided[:]
                PIG2['NeighborID'][start:end] = -1
                PIG2['NeighborID'][start:end][collided==1] = PIG2['Index'][start+nearest][:]

        if self.comm.rank == 0: self.logger.info("Finished fiber assignment")
    
        # return to the order specified by the global unique index
        mpsort.sort(PIG2, orderby='Index', out=PIG, comm=self.comm)
        
        # return arrays including the objects not in any groups
        collided = numpy.zeros(size[comm.rank], dtype='i4')
        collided[mask] = PIG['Collided'][:]
        neighbors = numpy.zeros(size[comm.rank], dtype='i4') - 1
        neighbors[mask] = PIG['NeighborID'][:]

        del PIG
        return collided, neighbors
    
    def _assign_multiplets(self, Position):
        """
        Internal function to assign the maximal amount of fibers 
        in collision groups of size N > 2
        """
        from scipy.spatial.distance import pdist, squareform
        
        def count(slice, n):
            return n[numpy.nonzero(slice)[0]].sum()
        
        # first shuffle the member ids, so we select random element when tied
        N = len(Position)
        group_ids = list(range(N))
    
        collided_ids = []
        while len(group_ids) > 1:
       
            # compute dists and find where dists < collision radius
            dists = squareform(pdist(Position[group_ids], metric='euclidean'))
            numpy.fill_diagonal(dists, numpy.inf) # ignore self-pairs
            collisions = dists <= self._collision_radius_rad
            
            # total # of collisions for each group member
            n_collisions = numpy.sum(collisions, axis=0)
            
            # total # of collisions for those objects that collide with each group member
            n_other = numpy.apply_along_axis(count, 0, collisions, n_collisions)
            
            # remove object that has most # of collisions 
            # and those colliding objects have least # of collisions
            idx = numpy.where(n_collisions == n_collisions.max())[0]
            
            # choose randomly, with a fixed local seed
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
        group_indices = list(range(N))
        dists = squareform(pdist(Position, metric='euclidean'))
        uncollided = [i for i in group_indices if i not in collided_ids]
        
        for i in sorted(collided_ids):
            neighbor = uncollided[dists[i][uncollided].argmin()]
            neighbor_ids.append(neighbor)
            
        collided = numpy.zeros(N)
        collided[collided_ids] = 1
        return collided, neighbor_ids
            
    def save(self, output, result):
        """
        Write the `Label`, `Collided`, and `NeighborID` arrays
        as a Pandas DataFrame to an HDF file, with key `FiberCollisonGroups`
        """
        import pandas as pd
        import os

        # gather the result to root and output
        result = self.comm.gather(result, root=0)
        
        if self.comm.rank == 0:
            
            # enforce a default extension
            _, ext = os.path.splitext(output)
            if 'hdf' not in ext: output += '.hdf5'
            
            self.logger.info("saving (Label, Collided, NeighborID) as Pandas HDF with name %s" %output)
        
            result = numpy.concatenate(result, axis=0)
            df = pd.DataFrame.from_records(result)
            df.to_hdf(output, 'FiberCollisionGroups')

