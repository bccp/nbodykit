from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_fibercolls(comm):
    
    from scipy.spatial.distance import pdist, squareform 
    from nbodykit.utils import ScatterArray, GatherArray
    
    CurrentMPIComm.set(comm)
    N = 10000
    
    # generate the initial data
    numpy.random.seed(42)
    if comm.rank == 0:
        ra = 10.*numpy.random.random(size=N)
        dec = 5.*numpy.random.random(size=N) - 5.0
    else:
        ra = None
        dec = None
    
    ra = ScatterArray(ra, comm)
    dec = ScatterArray(dec, comm)

    # compute the fiber collisions
    r = FiberCollisions(ra, dec, degrees=True, seed=42)
    rad = r._collision_radius_rad
    
    #  gather collided and position to root
    idx = GatherArray(r.labels['Collided'].compute().astype(bool), comm, root=0)
    pos = GatherArray(r.source['Position'].compute(), comm, root=0)
    
    # manually compute distances and check on root
    if comm.rank == 0:
        dists = squareform(pdist(pos, metric='euclidean'))
        numpy.fill_diagonal(dists, numpy.inf) # ignore self pairs
    
        # no objects in clean sample (Collided==0) should be within
        # the collision radius of any other objects in the sample
        clean_dists = dists[~idx, ~idx]
        assert (clean_dists <= rad).sum() == 0, "objects in 'clean' sample within collision radius!"
    
        # the collided objects must collided with at least 
        # one object in the clean sample
        ncolls_per = (dists[idx] <= rad).sum(axis=-1)
        assert (ncolls_per >= 1).all(), "objects in 'collided' sample that do not collide with any objects!"

    