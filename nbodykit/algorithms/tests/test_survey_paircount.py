from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import kdcount.correlate as correlate
import os
import pytest

# debug logging
setup_logging("debug")

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_data(seed):

    cosmo = cosmology.Planck15
    s = RandomCatalog(1000, seed=seed)

    # ra, dec, z
    s['Redshift'] = s.rng.normal(loc=0.5, scale=0.1, size=s.size)
    s['RA'] = s.rng.uniform(low=110, high=260, size=s.size)
    s['DEC'] = s.rng.uniform(low=-3.6, high=60., size=s.size)

    # position
    s['Position'] = transform.SkyToCartesion(s['RA'], s['DEC'], s['Redshift'], cosmo=cosmo)

    return s

def kcount_paircount(pos1, w1, redges, Nmu, pos2=None, w2=None):
    """
    Verify the paircount algorithm using kdcount
    """
    # get the trees
    tree1 = correlate.points(pos1, boxsize=None, weights=w1)
    if pos2 is None:
        tree2 = tree1
    else:
        tree2 = correlate.points(pos2, boxsize=None, weights=w2)

    # setup the binning
    if Nmu == 0:
        bins = correlate.RBinning(redges)
    else:
        bins = correlate.RmuBinning(redges, Nmu, observer=(0,0,0), mu_min=0., absmu=True)

    # do the paircount
    pc = correlate.paircount(tree1, tree2, bins, np=0, usefast=False, compute_mean_coords=True)

    ravg = pc.mean_centers if Nmu == 0 else pc.mean_centers[0]
    return numpy.nan_to_num(pc.pair_counts), numpy.nan_to_num(ravg), pc.sum1

@MPITest([1, 4])
def test_1d_auto(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # random particles
    source = generate_data(seed=42)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform(size=len(source))

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    r = SurveyDataPairCount('1d', source, redges, cosmo, weight='Weight')

    pos = gather_data(source, 'Position')
    w = gather_data(source, 'Weight')

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos, w, redges, 0)
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

@MPITest([1, 4])
def test_1d_cross(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # random particles
    source1 = generate_data(seed=42)
    source1['Weight'] = source1.rng.uniform(size=source1.size)
    source2 = generate_data(seed=84)
    source2['Weight'] = source2.rng.uniform(size=source2.size)

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the paircount
    r = SurveyDataPairCount('1d', source1, redges, cosmo, source2=source2)

    pos1 = gather_data(source1, 'Position')
    pos2 = gather_data(source2, 'Position')
    w1 = gather_data(source1, 'Weight')
    w2 = gather_data(source2, 'Weight')

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos1, w1, redges, 0, pos2=pos2, w2=w2)
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

@MPITest([1, 4])
def test_2d_auto(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # random particles
    source = generate_data(seed=42)
    source['Weight'] = source.rng.uniform(size=source.size)

    # make the bin edges
    redges = numpy.linspace(10, 1000., 10)
    Nmu = 10

    # do the weighted paircount
    r = SurveyDataPairCount('2d', source, redges, cosmo, weight='Weight', Nmu=10)

    pos = gather_data(source, 'Position')
    w = gather_data(source, 'Weight')

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos, w, redges, Nmu)
    #assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])


@MPITest([1, 4])
def test_2d_cross(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # random particles
    source1 = generate_data(seed=42)
    source1['Weight'] = source1.rng.uniform(size=source1.size)
    source2 = generate_data(seed=84)
    source2['Weight'] = source2.rng.uniform(size=source2.size)

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)
    Nmu = 10

    # do the paircount
    r = SurveyDataPairCount('2d', source1, redges, cosmo, source2=source2, Nmu=10)

    pos1 = gather_data(source1, 'Position')
    pos2 = gather_data(source2, 'Position')
    w1 = gather_data(source1, 'Weight')
    w2 = gather_data(source2, 'Weight')

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos1, w1, redges, Nmu, pos2=pos2, w2=w2)
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

    # test save
    r.save('paircount-test.json')
    r2 = SurveyDataPairCount.load('paircount-test.json')
    assert_array_equal(r.result.data, r2.result.data)

    if comm.rank == 0: os.remove('paircount-test.json')

@MPITest([1])
def test_missing_columns(comm):

    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # random particles
    source = generate_data(seed=42)

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # missing column
    with pytest.raises(ValueError):
        r = SurveyDataPairCount('1d', source, redges, cosmo, ra='BAD')
