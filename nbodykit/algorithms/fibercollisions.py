from nbodykit import CurrentMPIComm
from nbodykit.source.catalog import ArrayCatalog

from nbodykit.transform import SkyToUnitSphere
import numpy
import logging

class FiberCollisions(object):
    """
    Run an angular FOF algorithm to determine fiber collision
    groups from an input catalog, and then assign fibers such that
    the maximum amount of object receive a fiber.

    This amounts to determining the following population of objects:

    - population 1:
        the maximal "clean" sample of objects in which each object is not
        angularly collided with any other object in this subsample
    - population 2:
        the potentially-collided objects; these objects are those
        that are fiber collided + those that have been "resolved"
        due to multiple coverage in tile overlap regions

    Results are computed when the object is inititalized. See the documenation
    of :func:`~FiberCollisions.run` for the attributes storing the results.

    Parameters
    ----------
    ra : array_like
        the right ascension coordinate column
    dec : array_like
        the declination coordinate column
    collision_radius : float, optional
        the size of the angular collision radius (in degrees); default
        is 62 arcseconds
    seed : int, optional
        the random seed to use when determining which objects get fibers
    degrees : bool, optional
        set to `True` if the units of ``ra`` and ``dec`` are degrees

    References
    ----------
    - `Guo et al. 2010 <http://arxiv.org/abs/1111.6598>`_
    """
    logger = logging.getLogger('FiberCollisions')

    @CurrentMPIComm.enable
    def __init__(self, ra, dec, collision_radius=62/60./60., seed=None,
                    degrees=True, comm=None):

        # compute the pos
        ra = ArrayCatalog.make_column(ra)
        dec = ArrayCatalog.make_column(dec)
        pos = SkyToUnitSphere(ra, dec, degrees=degrees).compute()

        # make the source
        dt = numpy.dtype([('Position', (pos.dtype.str, 3))])
        pos = numpy.squeeze(pos.view(dtype=dt))
        source = ArrayCatalog(pos, BoxSize=numpy.array([2., 2., 2.]), comm=comm)

        self.source = source
        self.comm = source.comm

        # set the seed randomly if it is None
        if seed is None:
            if self.comm.rank == 0:
                seed = numpy.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)

        # save the attrs
        self.attrs = {}
        self.attrs['collision_radius'] = collision_radius
        self.attrs['seed'] = seed
        self.attrs['degrees'] = degrees

        # store collision radius in radians
        self._collision_radius_rad = numpy.deg2rad(collision_radius)
        if self.comm.rank == 0:
            self.logger.info("collision radius in degrees = %.4f" %collision_radius)

        # compute
        self.run()

    def run(self):
        """
        Run the fiber assignment algorithm. This attaches the following
        attribute to the object:

        - :attr:`labels`

        .. note::

            The :attr:`labels` attribute has a 1-to-1 correspondence with
            the rows in the input source.

        Attributes
        ----------
        labels: :class:`~nbodykit.source.catalog.array.ArrayCatalog`; size: :attr:`size`
            a CatalogSource that has the following columns:

            - Label :
                the group labels for each object in the input
                CatalogSource; label == 0 objects are not in a group
            - Collided :
                a flag array specifying which objects are collided, i.e., do
                not receive a fiber
            - NeighborID :
                for those objects that are collided, this gives the (global)
                index of the nearest neighbor on the sky (0-indexed) in
                the input catalog ``source``, else it is set to -1
        """
        from astropy.utils.misc import NumpyRNGContext
        from nbodykit.algorithms import FOF

        # angular FOF: labels gives the global group ID corresponding to each
        # object in Position on this rank
        fof = FOF(self.source, self._collision_radius_rad, 1, absolute=True)

        # assign the fibers (in parallel)
        with NumpyRNGContext(self.attrs['seed']):
            collided, neighbors = self._assign_fibers(fof.labels)

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
        d = list(zip(['Label', 'Collided', 'NeighborID'], [fof.labels, collided, neighbors]))
        dtype = numpy.dtype([(col, x.dtype) for col, x in d])
        result = numpy.empty(len(fof.labels), dtype=dtype)
        for col, x in d: result[col] = x

        # make a particle source
        self.labels = ArrayCatalog(result, comm=self.comm, **self.source.attrs)

    def _assign_fibers(self, Label):
        """
        Internal function to divide the data by collision group
        across ranks and assign fibers, such that the minimum
        number of objects are collided out of the survey
        """
        import mpsort
        from mpi4py import MPI

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
        size = self.comm.allgather(size)
        Ntot = sum(size)
        offset = sum(size[:self.comm.rank])
        PIG['Index'] = offset + numpy.where(mask == True)[0]
        del Label

        Position = self.source.compute(self.source['Position'])
        PIG['Position'] = Position[mask]
        del Position
        Ntot = self.comm.allreduce(len(mask))
        Nhalo = self.comm.allreduce(
            PIG['Label'].max() if len(PIG['Label']) > 0 else 0, op=MPI.MAX) + 1

        # now count number of particles per halo
        PIG['Rank'] = PIG['Label'] % self.comm.size
        cnt = numpy.bincount(PIG['Rank'], minlength=self.comm.size)
        Nlocal = self.comm.allreduce(cnt)[self.comm.rank]

        # sort by rank and then label
        PIG2 = numpy.empty(Nlocal, PIG.dtype)
        mpsort.sort(PIG, orderby='Rank', out=PIG2, comm=self.comm)
        assert (PIG2['Rank'] == self.comm.rank).all()
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
                if len(nearest):
                    PIG2['NeighborID'][start:end][collided] = PIG2['Index'][start+nearest][:]

        if self.comm.rank == 0: self.logger.info("Finished fiber assignment")

        # return to the order specified by the global unique index
        mpsort.sort(PIG2, orderby='Index', out=PIG, comm=self.comm)

        # return arrays including the objects not in any groups
        collided = numpy.zeros(size[self.comm.rank], dtype='i4')
        collided[mask] = PIG['Collided'][:]
        neighbors = numpy.zeros(size[self.comm.rank], dtype='i4') - 1
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

        collided = numpy.zeros(N, dtype=bool)
        collided[collided_ids] = True
        return collided, neighbor_ids
