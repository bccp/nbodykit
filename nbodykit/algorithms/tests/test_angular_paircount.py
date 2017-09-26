from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import os
import pytest

from kdcount import correlate
from kdcount import sphere

# debug logging
setup_logging()

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_data(seed):

    cosmo = cosmology.Planck15
    s = RandomCatalog(1000, seed=seed)

    # ra, dec
    s['RA'] = s.rng.uniform(low=50, high=260, size=s.size)
    s['DEC'] = s.rng.uniform(low=-10.6, high=60., size=s.size)

    return s

def kcount_paircount(pos1, w1, edges, pos2=None, w2=None):
    """
    Verify the paircount algorithm using kdcount
    """
    # get the trees
    tree1 = sphere.points(*pos1, boxsize=None, weights=w1)
    if pos2 is None:
        tree2 = tree1
    else:
        tree2 = sphere.points(*pos2, boxsize=None, weights=w2)

    # setup the binning
    bins = sphere.AngularBinning(edges)

    # do the paircount
    pc = correlate.paircount(tree1, tree2, bins, np=0, usefast=False, compute_mean_coords=True)

    return numpy.nan_to_num(pc.pair_counts), numpy.nan_to_num(pc.mean_centers), pc.sum1

@MPITest([1, 4])
def test_1d_auto(comm):
    CurrentMPIComm.set(comm)

    # random particles
    source = generate_data(seed=42)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform(size=len(source))

    # make the bin edges
    edges = numpy.linspace(0.001, 100.0, 10)

    # do the weighted paircount
    r = AngularPairCount(source, edges, weight='Weight', show_progress=False)

    # cannot compute theta=0
    with pytest.raises(ValueError):
        r = AngularPairCount(source, numpy.linspace(0, 1.0, 10))

    ra = gather_data(source, 'RA')
    dec = gather_data(source, 'DEC')
    w = gather_data(source, 'Weight')

    # verify with kdcount
    npairs, thetaavg, wsum = kcount_paircount([ra,dec], w, edges)
    #assert_allclose(thetaavg, r.result['theta'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

@MPITest([1, 4])
def test_1d_cross(comm):
    CurrentMPIComm.set(comm)

    # random particles
    first = generate_data(seed=42)
    first['Weight'] = first.rng.uniform(size=first.size)
    second = generate_data(seed=84)
    second['Weight'] = second.rng.uniform(size=second.size)

    # make the bin edges
    edges = numpy.linspace(0.001, 100.0, 10)

    # do the paircount
    r = AngularPairCount(first, edges, second=second)

    ra1 = gather_data(first, 'RA')
    ra2 = gather_data(second, 'RA')
    dec1 = gather_data(first, 'DEC')
    dec2 = gather_data(second, 'DEC')
    w1 = gather_data(first, 'Weight')
    w2 = gather_data(second, 'Weight')

    # verify with kdcount
    npairs, thetaavg, wsum = kcount_paircount([ra1,dec1], w1, edges, pos2=[ra2,dec2], w2=w2)
    #assert_allclose(thetaavg, r.result['theta'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

    # test save
    r.save('angular-paircount-test.json')
    r2 = AngularPairCount.load('angular-paircount-test.json')
    assert_array_equal(r.result.data, r2.result.data)

    if comm.rank == 0: os.remove('angular-paircount-test.json')


@MPITest([1])
def test_missing_columns(comm):
    CurrentMPIComm.set(comm)

    # random particles
    source = generate_data(seed=42)

    # make the bin edges
    edges = numpy.linspace(0.01, 10.0, 10)

    # missing column
    with pytest.raises(ValueError):
        r = AngularPairCount(source, edges, ra='BAD')
