from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_array_equal, assert_allclose
import kdcount.correlate as correlate
import os
import pytest

setup_logging()

def get_spherical_volume(source):
    pos = gather_data(source, "Position")
    rad = ((pos - 0.5*source.attrs['BoxSize'])**2).sum(axis=-1)**0.5
    keep = rad < 0.5 * source.attrs['BoxSize'].min()
    ra = gather_data(source, "RA")[keep]
    dec = gather_data(source, "DEC")[keep]
    return numpy.vstack([ra, dec]).T

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_sim_data(seed, comm):
    s = UniformCatalog(nbar=1000, BoxSize=1.0, seed=seed, comm=comm)
    s['RA'], s['DEC'] = transform.CartesianToEquatorial(s['Position'], observer=0.5*s.attrs['BoxSize'])
    return s

def reference_sim_tpcf(pos1, theta_edges, randoms=None, pos2=None):
    """Refernce angular 2PCF using halotools"""
    from halotools.mock_observables import angular_tpcf

    estimator = 'Natural' if randoms is None else 'Landy-Szalay'
    do_auto = True if pos2 is None else False
    return angular_tpcf(pos1, theta_edges, sample2=pos2, randoms=randoms,
                            estimator=estimator, do_auto=do_auto)

@MPITest([1, 4])
def test_sim_nonperiodic_auto(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)
    randoms = generate_sim_data(seed=84, comm=comm)

    # make the bin edges
    theta_edges = numpy.linspace(0.1, 10.0, 20)

    # compute 2PCF
    r = SimulationBox2PCF('angular', source, theta_edges, periodic=False, randoms1=randoms)

    # verify with reference
    cf = [-1.409027, 0.381019, -0.4193353, -0.1228877, -0.1578114, -0.1555974, 0.02935334, 0.04872244, 0.005932454,
        -0.3038669, 0.03336205, -0.07142007, -0.08075831, -0.07201585, -0.09670292, -0.1443989, -0.1554421, -0.2775648, 0.01500372]

    # verify with halotools // uses N*N instead of N*(N-1) normalization
    #ra1, dec1 = gather_data(source, "RA"),  gather_data(source, "DEC")
    #ra2, dec2 = gather_data(randoms, "RA"),  gather_data(randoms, "DEC")
    #cf = reference_sim_tpcf(numpy.vstack([ra1,dec1]).T, theta_edges, randoms=numpy.vstack([ra2,dec2]).T)
    assert_allclose(cf, r.corr['corr'], rtol=1e-5, atol=1e-3)

@MPITest([1, 4])
def test_sim_periodic_cross(comm):

    # uniform source of particles
    data1 = generate_sim_data(seed=42, comm=comm)
    data2 = generate_sim_data(seed=84, comm=comm)

    # make the bin edges
    theta_edges = numpy.linspace(0.1, 10.0, 20)

    # compute 2PCF
    r = SimulationBox2PCF('angular', data1, theta_edges, periodic=True, data2=data2)

    # verify with halotools
    sample1 = get_spherical_volume(data1)
    sample2 = get_spherical_volume(data2)
    cf = reference_sim_tpcf(sample1, theta_edges, pos2=sample2)
    assert_allclose(cf, r.corr['corr'])

@MPITest([1, 4])
def test_survey_auto(comm):

    # uniform source of particles
    data = generate_sim_data(seed=42, comm=comm)
    randoms = generate_sim_data(seed=84, comm=comm)

    # make the bin edges
    theta_edges = numpy.linspace(0.1, 10.0, 20)

    # compute 2PCF
    r = SurveyData2PCF('angular', data, randoms, theta_edges)

    # verify with reference
    cf = [-1.409027, 0.381019, -0.4193353, -0.1228877, -0.1578114, -0.1555974, 0.02935334, 0.04872244, 0.005932454,
        -0.3038669, 0.03336205, -0.07142007, -0.08075831, -0.07201585, -0.09670292, -0.1443989, -0.1554421, -0.2775648, 0.01500372]

    # verify with halotools // uses N*N instead of N*(N-1) normalization
    #ra1, dec1 = gather_data(data, "RA"),  gather_data(data, "DEC")
    #ra2, dec2 = gather_data(randoms, "RA"),  gather_data(randoms, "DEC")
    #cf = reference_sim_tpcf(numpy.vstack([ra1,dec1]).T, theta_edges, randoms=numpy.vstack([ra2,dec2]).T)
    assert_allclose(cf, r.corr['corr'], rtol=1e-5, atol=1e-3)
