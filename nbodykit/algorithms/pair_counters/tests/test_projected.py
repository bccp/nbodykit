from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
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

def reference_sim_paircount(pos1, redges, pimax, boxsize, pos2=None, los=2):
    """Reference pair counting via halotools"""
    from halotools.mock_observables.pair_counters import npairs_xy_z

    # reorder the axes
    axes_order = [i for i in [0,1,2] if i != los] + [los]
    pos1 = pos1[:,axes_order]

    # the pi bins
    pibins = numpy.linspace(0, pimax, pimax+1)

    if pos2 is None:
        pos2 = pos1
    else:
        pos2 = pos2[:,axes_order]
    r = npairs_xy_z(pos1, pos2, redges, pibins, period = boxsize)
    return numpy.diff(numpy.diff(r, axis=0), axis=1)

def reference_survey_paircount(pos1, w1, rp_bins, pimax, pos2=None, w2=None, los=2):
    """Reference pair counting via kdcount"""
    from kdcount import KDTree, correlate

    pi_bins = numpy.linspace(0, pimax, int(pimax)+1)
    tree1 = correlate.points(pos1, boxsize=None, weights=w1).tree.root
    if pos2 is None:
        pos2 = pos1; w2 = w1
    tree2 = correlate.points(pos2, boxsize=None, weights=w2).tree.root

    if w1 is None:
        w1 = numpy.ones_like(pos1)
    if w2 is None:
        w2 = numpy.ones_like(pos2)

    # find all pairs
    r, i, j = tree1.enum(tree2, (pimax**2 + rp_bins[-1]**2)**0.5)

    def compute_rp_pi(r, i, j):
        r1 = pos1[i]
        r2 = pos2[j]
        center = 0.5 * (r1 + r2)
        dr = r1 - r2
        dot = numpy.einsum('ij, ij->i', dr, center)
        dot2 = dot ** 2
        center2 = numpy.einsum('ij, ij->i', center, center)
        los2 = dot2 / center2
        dr2 = numpy.einsum('ij, ij->i', dr, dr)
        x2 = numpy.abs(dr2 - los2)
        return x2 ** 0.5, los2 ** 0.5

    # compute rp, pi distances
    rp, pi = compute_rp_pi(r, i, j)

    # digitize
    rp_dig = numpy.digitize(rp, rp_bins)
    pi_dig = numpy.digitize(pi, pi_bins)
    shape = (len(rp_bins)+1,len(pi_bins)+1)
    multi_index = numpy.ravel_multi_index([rp_dig, pi_dig], shape)

    # initialize the return arrays
    npairs = numpy.zeros(shape, dtype='i8')
    rpavg = numpy.zeros(shape, dtype='f8')
    wnpairs = numpy.zeros(shape, dtype='f8')

    # mean rp values
    rpavg.flat += numpy.bincount(multi_index, weights=rp, minlength=rpavg.size)
    rpavg = rpavg[1:-1,1:-1]

    # count the pairs
    npairs.flat += numpy.bincount(multi_index, minlength=npairs.size)
    npairs = npairs[1:-1,1:-1]

    wnpairs.flat += numpy.bincount(multi_index, weights=w1[i]*w2[j], minlength=wnpairs.size)
    wnpairs = wnpairs[1:-1,1:-1]

    return npairs, numpy.nan_to_num(rpavg/npairs), numpy.nan_to_num(wnpairs)


@MPITest([1, 3])
def test_sim_periodic_auto(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform()

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    pimax = 40.
    r = SimulationBoxPairCount('projected', source, redges, periodic=True, weight='Weight', pimax=pimax)

    pos = gather_data(source, "Position")

    # verify with halotools
    npairs = reference_sim_paircount(pos, redges, pimax, source.attrs['BoxSize'])
    assert_allclose(npairs, r.pairs['npairs'])

@MPITest([4])
def test_sim_diff_los(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)

    # add some weights b/w 0 and 1
    source['Weight'] = source.rng.uniform()

    # make the bin edges
    redges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    pimax = 100.
    r = SimulationBoxPairCount('projected', source, redges, periodic=True, weight='Weight', pimax=pimax, los='x')

    # verify with halotools
    pos = gather_data(source, "Position")
    npairs = reference_sim_paircount(pos, redges, pimax, source.attrs['BoxSize'], los=0)
    assert_allclose(npairs, r.pairs['npairs'])

@MPITest([1, 3])
def test_sim_nonperiodic_auto(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)
    source['Weight'] = numpy.random.random(size=len(source))

    # make the bin edges
    redges = numpy.linspace(10, 250., 10)

    # do the weighted paircount
    pimax = 50.
    r = SimulationBoxPairCount('projected', source, redges, periodic=False, weight='Weight', pimax=pimax)

    # verify with halotools
    pos = gather_data(source, "Position")
    npairs = reference_sim_paircount(pos, redges, pimax, None)
    assert_allclose(npairs, r.pairs['npairs'])


@MPITest([1, 3])
def test_sim_periodic_cross(comm):

    # generate data
    first = generate_sim_data(seed=42, comm=comm)
    second = generate_sim_data(seed=84, comm=comm)

    # make the bin edges
    redges = numpy.linspace(10, 40., 10)

    # do the paircount
    pimax = 10.
    r = SimulationBoxPairCount('projected', first, redges, second=second, periodic=True, pimax=pimax)

    pos1 = gather_data(first, "Position")
    pos2 = gather_data(second, "Position")

    # verify with halotools
    npairs = reference_sim_paircount(pos1, redges, pimax, first.attrs['BoxSize'], pos2=pos2)
    assert_allclose(npairs, r.pairs['npairs'])

@MPITest([1, 4])
def test_survey_auto(comm):
    cosmo = cosmology.Planck15

    # random particles
    source = generate_survey_data(seed=42, comm=comm)
    source['Weight'] = source.rng.uniform()

    # make the bin edges
    redges = numpy.linspace(10, 1000., 10)

    # do the weighted paircount
    pimax = 500.
    r = SurveyDataPairCount('projected', source, redges, cosmo, weight='Weight', pimax=pimax)

    pos = gather_data(source, 'Position')
    w = gather_data(source, 'Weight')

    # verify with kdcount
    npairs, ravg, wnpairs = reference_survey_paircount(pos, w, redges, pimax)
    assert_allclose(ravg, r.pairs['rp'], rtol=1e-5)
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wnpairs, r.pairs['wnpairs'])

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
    pimax = 50.
    r = SurveyDataPairCount('projected', first, redges, cosmo, second=second, pimax=pimax)

    pos1 = gather_data(first, 'Position')
    pos2 = gather_data(second, 'Position')
    w1 = gather_data(first, 'Weight')
    w2 = gather_data(second, 'Weight')

    # verify with kdcount
    npairs, ravg, wnpairs = reference_survey_paircount(pos1, w1, redges, pimax, pos2=pos2, w2=w2)
    assert_allclose(ravg, r.pairs['rp'], rtol=1e-3)
    assert_allclose(npairs, r.pairs['npairs'])
    assert_allclose(wnpairs, r.pairs['wnpairs'])

    # test save
    r.save('paircount-test.json')
    r2 = SurveyDataPairCount.load('paircount-test.json', comm=comm)
    assert_array_equal(r.pairs.data, r2.pairs.data)

    if comm.rank == 0: os.remove('paircount-test.json')

@MPITest([1])
def test_missing_pimax(comm):

    # generate data
    source = generate_sim_data(seed=42, comm=comm)
    redges = numpy.linspace(10, 150, 10)

    # missing pimax
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('projected', source, redges)

    # wrong mode
    with pytest.raises(ValueError):
        r = SimulationBoxPairCount('1d', source, redges, pimax=10.)

@MPITest([1])
def test_bad_pimax(comm):

    # generate data
    source = generate_sim_data(seed=42, comm=comm)

    # pimax must be at least 1
    with pytest.raises(ValueError):
        redges = numpy.linspace(10, 150, 10)
        r = SimulationBoxPairCount('projected', source, redges, pimax=0.5)
