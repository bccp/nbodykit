from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_array_equal, assert_allclose
import kdcount.correlate as correlate
import os
import pytest

setup_logging()
data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')

def get_spherical_volume(source):
    pos = gather_data(source, "Position")
    rad = ((pos - 0.5*source.attrs['BoxSize'])**2).sum(axis=-1)**0.5
    keep = rad < 0.5 * source.attrs['BoxSize'].min()
    ra = gather_data(source, "RA")[keep]
    dec = gather_data(source, "DEC")[keep]
    return numpy.vstack([ra, dec]).T

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_sim_data(seed):
    s = UniformCatalog(nbar=1000, BoxSize=1.0, seed=seed)
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
    CurrentMPIComm.set(comm)

    # uniform source of particles
    BoxSize = 1.
    source = CSVCatalog(os.path.join(data_dir,'test_angular_sim_data.dat'),names=['x', 'y', 'z'])
    source['Position'] = transform.StackColumns(source['x'], source['y'], source['z'])
    randoms = CSVCatalog(os.path.join(data_dir,'test_angular_sim_randoms.dat'),names=['x', 'y', 'z'])
    randoms['Position'] = transform.StackColumns(randoms['x'], randoms['y'], randoms['z'])

    # make the bin edges
    theta_edges = numpy.linspace(0.1, 10.0, 20)

    # compute 2PCF
    r = SimulationBox2PCF('angular', source, theta_edges, periodic=False, BoxSize=BoxSize, randoms1=randoms)

    # verify with reference
    cf = numpy.loadtxt(os.path.join(data_dir,'test_angular_sim_nonperiodic_auto.dat'))
    assert_allclose(cf, r.corr['corr'], rtol=1e-5, atol=1e-3)

@MPITest([1, 4])
def test_sim_periodic_cross(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    data1 = generate_sim_data(seed=42)
    data2 = generate_sim_data(seed=84)

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
    CurrentMPIComm.set(comm)

    # uniform source of particles

    data = CSVCatalog(os.path.join(data_dir,'test_angular_survey_data.dat'),names=['RA', 'DEC'])
    randoms = CSVCatalog(os.path.join(data_dir,'test_angular_survey_randoms.dat'),names=['RA', 'DEC'])

    # make the bin edges
    theta_edges = numpy.linspace(0.1, 10.0, 20)

    # compute 2PCF
    r = SurveyData2PCF('angular', data, randoms, theta_edges)

    # load reference
    DD, DR, _, RR, cf = numpy.loadtxt(os.path.join(data_dir,'test_angular_survey_auto.dat'),unpack=True)

    # verify pair counts and CF
    assert_allclose(cf, r.corr['corr'])
    assert_allclose(DD, r.D1D2['npairs'])
    assert_allclose(DR, r.D1R2['npairs'])
    assert_allclose(RR, r.R1R2['npairs'])
