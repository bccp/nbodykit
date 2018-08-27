from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
import kdcount.correlate as correlate
import os
import pytest

setup_logging()

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_sim_data(seed, comm):
    return UniformCatalog(nbar=3e-6, BoxSize=512., seed=seed, comm=comm)

def generate_survey_data(seed, comm):
    cosmo = cosmology.Planck15
    s = RandomCatalog(1000, seed=seed, comm=comm)

    # ra, dec, z
    s['Redshift'] = s.rng.normal(loc=0.5, scale=0.1)
    s['RA'] = s.rng.uniform(low=110, high=260)
    s['DEC'] = s.rng.uniform(low=-3.6, high=60.)

    # position
    s['Position'] = transform.SkyToCartesian(s['RA'], s['DEC'], s['Redshift'], cosmo=cosmo)

    return s

def reference_sim_paircount(pos1, w1, redges, Nmu, boxsize, pos2=None, w2=None, los=2):
    """Reference pair counting via kdcount"""

    tree1 = correlate.points(pos1, boxsize=boxsize, weights=w1)
    if pos2 is None:
        tree2 = tree1
    else:
        tree2 = correlate.points(pos2, boxsize=boxsize, weights=w2)

    bins = correlate.FlatSkyBinning(redges, Nmu, los=los, mu_min=0., absmu=True,)
    pc = correlate.paircount(tree1, tree2, bins, np=0, usefast=False, compute_mean_coords=True)
    return numpy.nan_to_num(pc.pair_counts), numpy.nan_to_num(pc.mean_centers[0]), pc.sum1

def reference_survey_paircount(pos1, w1, redges, Nmu, pos2=None, w2=None, los=2):
    """Reference pair counting via kdcount"""

    tree1 = correlate.points(pos1, boxsize=None, weights=w1)
    if pos2 is None:
        tree2 = tree1
    else:
        tree2 = correlate.points(pos2, boxsize=None, weights=w2)

    bins = correlate.RmuBinning(redges, Nmu, observer=(0,0,0), mu_min=0., absmu=True)
    pc = correlate.paircount(tree1, tree2, bins, np=0, usefast=False, compute_mean_coords=True)
    return numpy.nan_to_num(pc.pair_counts), numpy.nan_to_num(pc.mean_centers[0]), pc.sum1


@MPITest([1, 3])
def test_sim_periodic_auto(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform()

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    Nmu = 10
    r = SimulationBoxPairCount('2d', source, redges, periodic=True, weight='Weight', Nmu=Nmu)

    pos = gather_data(source, "Position")
    w = gather_data(source, "Weight")

    # verify with kdcount
    npairs, ravg, wsum = reference_sim_paircount(pos, w, redges, Nmu, source.attrs['BoxSize'])
    assert_allclose(ravg, r.pairs['r'])
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])

@MPITest([3])
def test_sim_diff_los(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform()

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    Nmu = 2
    r = SimulationBoxPairCount('2d', source, redges, periodic=True, weight='Weight', Nmu=Nmu, los='x')

    pos = gather_data(source, "Position")
    w = gather_data(source, "Weight")

    # verify with kdcount
    npairs, ravg, wsum = reference_sim_paircount(pos, w, redges, Nmu, source.attrs['BoxSize'], los=0)
    assert_allclose(ravg, r.pairs['r'])
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])

@MPITest([1, 3])
def test_sim_nonperiodic_auto(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)
    source['Weight'] = numpy.random.random(size=len(source))

    # make the bin edges
    redges = numpy.linspace(10, 40., 10)

    # do the weighted paircount
    Nmu = 50
    r = SimulationBoxPairCount('2d', source, redges, periodic=False, weight='Weight', Nmu=Nmu)

    pos = gather_data(source, "Position")
    w = gather_data(source, "Weight")

    # verify with kdcount
    npairs, ravg, wsum = reference_sim_paircount(pos, w, redges, Nmu, None)
    assert_allclose(ravg, r.pairs['r'])
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])


@MPITest([1, 3])
def test_sim_periodic_cross(comm):

    # generate data
    first = generate_sim_data(seed=42, comm=comm)
    second = generate_sim_data(seed=84, comm=comm)

    # make the bin edges
    redges = numpy.linspace(10, 40., 10)

    # do the paircount
    Nmu = 10
    r = SimulationBoxPairCount('2d', first, redges, second=second, periodic=True, Nmu=Nmu)

    pos1 = gather_data(first, "Position")
    pos2 = gather_data(second, "Position")

    # verify with kdcount
    npairs, ravg, wsum = reference_sim_paircount(pos1, None, redges, Nmu, first.attrs['BoxSize'], pos2=pos2)
    assert_allclose(ravg, r.pairs['r'])
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])

@MPITest([1, 4])
def test_survey_auto(comm):
    cosmo = cosmology.Planck15

    # random particles
    source = generate_survey_data(seed=42, comm=comm)
    source['Weight'] = source.rng.uniform()

    # make the bin edges
    redges = numpy.linspace(10, 1000., 10)

    # do the weighted paircount
    Nmu = 50
    r = SurveyDataPairCount('2d', source, redges, cosmo, weight='Weight', Nmu=Nmu)

    pos = gather_data(source, 'Position')
    w = gather_data(source, 'Weight')

    # verify with kdcount
    npairs, ravg, wsum = reference_survey_paircount(pos, w, redges, Nmu)
    assert_allclose(ravg, r.pairs['r'])
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])


@MPITest([1, 4])
def test_survey_cross(comm):
    cosmo = cosmology.Planck15

    # random particles
    first = generate_survey_data(seed=42, comm=comm)
    first['Weight'] = first.rng.uniform()
    second = generate_survey_data(seed=84, comm=comm)
    second['Weight'] = second.rng.uniform()

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the paircount
    Nmu = 5
    r = SurveyDataPairCount('2d', first, redges, cosmo, second=second, Nmu=Nmu)

    pos1 = gather_data(first, 'Position')
    pos2 = gather_data(second, 'Position')
    w1 = gather_data(first, 'Weight')
    w2 = gather_data(second, 'Weight')

    # verify with kdcount
    npairs, ravg, wsum = reference_survey_paircount(pos1, w1, redges, Nmu, pos2=pos2, w2=w2)
    assert_allclose(ravg, r.pairs['r'])
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])

    # test save
    r.save('paircount-test.json')
    r2 = SurveyDataPairCount.load('paircount-test.json', comm=comm)
    assert_array_equal(r.pairs.data, r2.pairs.data)

    if comm.rank == 0: os.remove('paircount-test.json')

@MPITest([1])
def test_missing_Nmu(comm):

    # generate data
    source = generate_sim_data(seed=42, comm=comm)
    redges = numpy.linspace(10, 150, 10)

    # missing Nmu
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('2d', source, redges)

    # wrong mode
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('1d', source, redges, Nmu=10)
