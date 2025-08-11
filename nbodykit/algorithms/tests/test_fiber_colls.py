from nbodykit.lab import *
from nbodykit import setup_logging
from nbodykit.utils import ScatterArray, GatherArray
from numpy.testing import assert_array_equal
from mpi4py import MPI
import pytest
# debug logging
setup_logging("debug")

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_fibercolls(comm):

    from scipy.spatial.distance import pdist, squareform


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
    r = FiberCollisions(ra, dec, degrees=True, seed=42, comm=comm)
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
@pytest.mark.xfail
def test_fibercolls_issue584_4pt(comm):
    ra = numpy.array([0.,1.,2., 10])
    dec = numpy.array([0.,0.,0., 0])
    labels = FiberCollisions(ra,dec,collision_radius=1.5,seed=None, comm=comm).labels
    assert_array_equal(labels['Collided'].compute(), [0, 1, 0, 0])

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
@pytest.mark.xfail
def test_fibercolls_issue584_3pt(comm):
    ra = numpy.array([0.,1.,2.])
    dec = numpy.array([0.,0.,0.])
    labels = FiberCollisions(ra,dec,collision_radius=1.5,seed=None, comm=comm).labels
    assert_array_equal(labels['Collided'].compute(), [0, 1, 0])

