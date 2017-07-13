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

def kcount_paircount(pos1, w1, redges, Nmu, boxsize, pos2=None, w2=None, los=2):
    """
    Verify the paircount algorithm using kdcount
    """
    # get the trees
    tree1 = correlate.points(pos1, boxsize=boxsize, weights=w1)
    if pos2 is None:
        tree2 = tree1
    else:
        tree2 = correlate.points(pos2, boxsize=boxsize, weights=w2)

    # setup the binning
    if Nmu == 0:
        bins = correlate.RBinning(redges)
    else:
        bins = correlate.FlatSkyBinning(redges, Nmu, los=los, mu_min=0., absmu=True,)

    # do the paircount
    pc = correlate.paircount(tree1, tree2, bins, np=0, usefast=False, compute_mean_coords=True)

    ravg = pc.mean_centers if Nmu == 0 else pc.mean_centers[0]
    return numpy.nan_to_num(pc.pair_counts), numpy.nan_to_num(ravg), pc.sum1

@MPITest([1, 3])
def test_1d_periodic_auto(comm):

    CurrentMPIComm.set(comm)

    # uniform source of particles
    BoxSize = 512.
    source = UniformCatalog(nbar=3e-6, BoxSize=BoxSize, seed=42)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform(size=len(source))

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    r = SimulationBoxPairCount('1d', source, redges, periodic=True, weight='Weight')

    pos = gather_data(source, "Position")
    w = gather_data(source, "Weight")

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos, w, redges, 0, source.attrs['BoxSize'])
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

    # test save
    r.save('paircount-test.json')
    r2 = SimulationBoxPairCount.load('paircount-test.json')
    assert_array_equal(r.result.data, r2.result.data)

    if comm.rank == 0: os.remove('paircount-test.json')

@MPITest([1, 3])
def test_1d_nonperiodic_auto(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    BoxSize = 512.
    source = UniformCatalog(nbar=3e-6, BoxSize=BoxSize, seed=42)
    source['Weight'] = numpy.random.random(size=len(source))

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    r = SimulationBoxPairCount('1d', source, redges, periodic=False, weight='Weight')

    pos = gather_data(source, "Position")
    w = gather_data(source, "Weight")

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos, w, redges, 0, None)
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])


@MPITest([1, 3])
def test_1d_periodic_cross(comm):

    CurrentMPIComm.set(comm)

    BoxSize = 512.
    source1 = UniformCatalog(nbar=3e-6, BoxSize=BoxSize, seed=42)
    source2 = UniformCatalog(nbar=3e-6, BoxSize=BoxSize, seed=84)

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the paircount
    r = SimulationBoxPairCount('1d', source1, redges, source2=source2, periodic=True)

    pos1 = gather_data(source1, "Position")
    pos2 = gather_data(source2, "Position")

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos1, None, redges, 0, source1.attrs['BoxSize'], pos2=pos2)
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

@MPITest([1, 3])
def test_2d_periodic_auto(comm):

    CurrentMPIComm.set(comm)

    # uniform source of particles
    BoxSize = 512.
    source = UniformCatalog(nbar=3e-6, BoxSize=BoxSize, seed=42)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform(size=len(source))

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)
    Nmu = 10

    # do the weighted paircount
    r = SimulationBoxPairCount('2d', source, redges, periodic=True, weight='Weight', Nmu=10)

    pos = gather_data(source, "Position")
    w = gather_data(source, "Weight")

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos, w, redges, Nmu, source.attrs['BoxSize'])
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

@MPITest([3])
def test_2d_diff_los(comm):

    CurrentMPIComm.set(comm)

    # uniform source of particles
    BoxSize = 512.
    source = UniformCatalog(nbar=3e-6, BoxSize=BoxSize, seed=42)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform(size=len(source))

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)
    Nmu = 10

    # do the weighted paircount
    r = SimulationBoxPairCount('2d', source, redges, periodic=True, weight='Weight', Nmu=10, los='x')

    pos = gather_data(source, "Position")
    w = gather_data(source, "Weight")

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos, w, redges, Nmu, source.attrs['BoxSize'], los=0)
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

@MPITest([1, 3])
def test_2d_nonperiodic_auto(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    source = UniformCatalog(nbar=3e-6, BoxSize=512, seed=42)
    source['Weight'] = numpy.random.random(size=len(source))

    # make the bin edges
    redges = numpy.linspace(10, 40., 10)
    Nmu = 10

    # do the weighted paircount
    r = SimulationBoxPairCount('2d', source, redges, periodic=False, weight='Weight', Nmu=10)

    pos = gather_data(source, "Position")
    w = gather_data(source, "Weight")

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos, w, redges, Nmu, None)
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])


@MPITest([1, 3])
def test_2d_periodic_cross(comm):

    CurrentMPIComm.set(comm)

    BoxSize = 512.
    source1 = UniformCatalog(nbar=3e-6, BoxSize=BoxSize, seed=42)
    source2 = UniformCatalog(nbar=3e-6, BoxSize=BoxSize, seed=84)

    # make the bin edges
    redges = numpy.linspace(10, 40., 10)
    Nmu = 10

    # do the paircount
    r = SimulationBoxPairCount('2d', source1, redges, source2=source2, periodic=True, Nmu=10)

    pos1 = gather_data(source1, "Position")
    pos2 = gather_data(source2, "Position")

    # verify with kdcount
    npairs, ravg, wsum = kcount_paircount(pos1, None, redges, Nmu, source1.attrs['BoxSize'], pos2=pos2)
    assert_allclose(ravg, r.result['r'])
    assert_allclose(npairs, r.result['npairs'])
    assert_allclose(wsum, r.result['npairs'] * r.result['weightavg'])

@MPITest([1])
def test_bad_los(comm):

    CurrentMPIComm.set(comm)

    # uniform source of particles
    source = UniformCatalog(nbar=3e-6, BoxSize=512., seed=42)

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # should be 'x', 'y', 'z'
    with pytest.raises(AssertionError):
        r = SimulationBoxPairCount('1d', source, redges, los='a')

    # should be [0,1,2]
    with pytest.raises(AssertionError):
        r = SimulationBoxPairCount('1d', source, redges, los=3)

    # negative okay
    r = SimulationBoxPairCount('1d', source, redges, los=-1)

    # vector is bad
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('1d', source, redges, los=[0,0,1])

@MPITest([1])
def test_bad_rmax(comm):

    CurrentMPIComm.set(comm)

    # uniform source of particles
    source = UniformCatalog(nbar=3e-6, BoxSize=512., seed=42)

    # make the bin edges
    redges = numpy.linspace(10, 400., 10)

    # rmax is too big
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('1d', source, redges, periodic=True)

@MPITest([1])
def test_noncubic_box(comm):

    CurrentMPIComm.set(comm)

    # uniform source of particles
    source = UniformCatalog(nbar=3e-6, BoxSize=[512., 512., 256], seed=42)

    # make the bin edges
    redges = numpy.linspace(10, 100., 10)

    # cannot do non-cubic boxes
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('1d', source, redges, periodic=True)

@MPITest([1])
def test_bad_boxsize(comm):

    CurrentMPIComm.set(comm)

    # uniform source of particles
    source1 = UniformCatalog(nbar=3e-5, BoxSize=512.0, seed=42)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=256.0, seed=42)

    # make the bin edges
    redges = numpy.linspace(10, 50., 10)

    # box sizes are different
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('1d', source1, redges, source2=source2)

    # specified different value
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('1d', source2, redges, BoxSize=300.)

    # no boxSize
    del source1.attrs['BoxSize']
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('1d', source1, redges)
