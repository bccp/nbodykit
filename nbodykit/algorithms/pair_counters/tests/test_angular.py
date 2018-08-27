from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

import os
import pytest
import numpy
from numpy.testing import assert_array_equal, assert_allclose
from kdcount import correlate, sphere

# debug logging
setup_logging()

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_survey_data(seed, comm):
    s = RandomCatalog(1000, seed=seed, comm=comm)
    s['RA'] = s.rng.uniform(low=50, high=260)
    s['DEC'] = s.rng.uniform(low=-10.6, high=60.)
    return s

def generate_sim_data(seed, comm):
    s = UniformCatalog(nbar=1000, BoxSize=1.0, seed=seed, comm=comm)
    s['RA'], s['DEC'] = transform.CartesianToEquatorial(s['Position'], observer=0.5*s.attrs['BoxSize'])
    return s

def reference_paircount(pos1, w1, edges, pos2=None, w2=None):
    """Reference pair counting via kdcount"""
    # set up the trees
    tree1 = sphere.points(*pos1, boxsize=None, weights=w1)
    if pos2 is None:
        tree2 = tree1
    else:
        tree2 = sphere.points(*pos2, boxsize=None, weights=w2)

    # run the pair count
    bins = sphere.AngularBinning(edges)
    pc = correlate.paircount(tree1, tree2, bins, np=0, usefast=False, compute_mean_coords=True)

    return numpy.nan_to_num(pc.pair_counts), numpy.nan_to_num(pc.mean_centers), pc.sum1

@MPITest([1, 4])
def test_survey_auto(comm):

    # random particles
    source = generate_survey_data(seed=42, comm=comm)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform()

    # make the bin edges
    edges = numpy.linspace(0.001, 1.0, 10)

    # do the weighted paircount
    r = SurveyDataPairCount('angular', source, edges, weight='Weight', show_progress=False)

    # cannot compute theta=0
    with pytest.raises(ValueError):
        r = SurveyDataPairCount('angular', source, numpy.linspace(0, 1.0, 10))

    ra = gather_data(source, 'RA')
    dec = gather_data(source, 'DEC')
    w = gather_data(source, 'Weight')

    # verify with kdcount
    npairs, thetaavg, wsum = reference_paircount([ra,dec], w, edges)
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])

@MPITest([1, 4])
def test_survey_cross(comm):

    # random particles with weights
    first = generate_survey_data(seed=42, comm=comm)
    first['Weight'] = first.rng.uniform()
    second = generate_survey_data(seed=84, comm=comm)
    second['Weight'] = second.rng.uniform()

    # make the bin edges
    edges = numpy.linspace(0.001, 1.0, 10)

    # do the paircount
    r = SurveyDataPairCount('angular', first, edges, second=second)

    # gather data to run kdcount
    ra1 = gather_data(first, 'RA')
    ra2 = gather_data(second, 'RA')
    dec1 = gather_data(first, 'DEC')
    dec2 = gather_data(second, 'DEC')
    w1 = gather_data(first, 'Weight')
    w2 = gather_data(second, 'Weight')

    # verify with kdcount
    npairs, thetaavg, wsum = reference_paircount([ra1,dec1], w1, edges, pos2=[ra2,dec2], w2=w2)
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])

    # test save
    r.save('angular-paircount-test.json')
    r2 = SurveyDataPairCount.load('angular-paircount-test.json', comm=comm)
    assert_array_equal(r.pairs.data, r2.pairs.data)

    if comm.rank == 0: os.remove('angular-paircount-test.json')

@MPITest([1, 4])
def test_sim_auto(comm):

    # random particles
    source = generate_sim_data(seed=42, comm=comm)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform()

    # make the bin edges
    edges = numpy.linspace(0.001, 1.0, 10)

    # do the weighted paircount
    r = SimulationBoxPairCount('angular', source, edges, weight='Weight', show_progress=False)

    # cannot compute theta=0
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('angular', source, numpy.linspace(0, 1.0, 10))

    ra = gather_data(source, 'RA')
    dec = gather_data(source, 'DEC')
    w = gather_data(source, 'Weight')

    # verify with kdcount
    npairs, thetaavg, wsum = reference_paircount([ra,dec], w, edges)
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])

@MPITest([1, 4])
def test_sim_cross(comm):

    # random particles with weights
    first = generate_sim_data(seed=42, comm=comm)
    first['Weight'] = first.rng.uniform()
    second = generate_sim_data(seed=84, comm=comm)
    second['Weight'] = second.rng.uniform()

    # make the bin edges
    edges = numpy.linspace(0.001, 1.0, 10)

    # do the paircount
    r = SimulationBoxPairCount('angular', first, edges, second=second)

    # gather data to run kdcount
    ra1, ra2 = gather_data(first, 'RA'), gather_data(second, 'RA')
    dec1, dec2 = gather_data(first, 'DEC'), gather_data(second, 'DEC')
    w1, w2 = gather_data(first, 'Weight'), gather_data(second, 'Weight')

    # verify with kdcount
    npairs, thetaavg, wsum = reference_paircount([ra1,dec1], w1, edges, pos2=[ra2,dec2], w2=w2)
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wsum, r.pairs['wnpairs'])

    # test save
    r.save('angular-paircount-test.json')
    r2 = SimulationBoxPairCount.load('angular-paircount-test.json', comm=comm)
    assert_array_equal(r.pairs.data, r2.pairs.data)

    if comm.rank == 0: os.remove('angular-paircount-test.json')


@MPITest([1])
def test_missing_columns(comm):

    source = generate_survey_data(seed=42, comm=comm)
    edges = numpy.linspace(0.01, 10.0, 10)

    # missing column
    with pytest.raises(ValueError):
        r = SurveyDataPairCount('angular', source, edges, ra='BAD')
