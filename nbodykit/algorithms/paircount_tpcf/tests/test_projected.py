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

def generate_sim_data(seed):
    return UniformCatalog(nbar=3e-4, BoxSize=512., seed=seed)

def reference_sim_tpcf(pos1, redges, pimax, BoxSize, randoms=None, pos2=None):
    """Refernce projected 2PCF using halotools"""
    from halotools.mock_observables import rp_pi_tpcf
    
    pi_bins = numpy.linspace(0, pimax, int(pimax)+1)
    estimator = 'Natural' if randoms is None else 'Landy-Szalay'
    do_auto = True if pos2 is None else False
    return rp_pi_tpcf(pos1, redges, pi_bins, period=BoxSize, sample2=pos2,
                    randoms=randoms, estimator=estimator, do_auto=do_auto)

@MPITest([1, 3])
def test_sim_periodic_auto(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    source = generate_sim_data(seed=42)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    pimax = 50.
    r = SimulationBox2PCF('projected', source, redges, pimax=pimax, periodic=True)

    # verify with halotools
    pos = gather_data(source, "Position")
    cf = reference_sim_tpcf(pos, redges, pimax, source.attrs['BoxSize'])
    assert_allclose(cf, r.corr['corr'])

@MPITest([1, 3])
def test_sim_nonperiodic_auto(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    source = generate_sim_data(seed=42)
    randoms = generate_sim_data(seed=84)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    pimax = 50.
    r = SimulationBox2PCF('projected', source, redges, periodic=False, pimax=pimax, randoms1=randoms)

    # verify with halotools
    pos_d = gather_data(source, "Position")
    pos_r = gather_data(randoms, "Position")
    cf = reference_sim_tpcf(pos_d, redges, pimax, None, randoms=pos_r)
    assert_allclose(cf, r.corr['corr'], rtol=1e-5, atol=1e-3)

@MPITest([1, 3])
def test_sim_periodic_cross(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    data1 = generate_sim_data(seed=42)
    data2 = generate_sim_data(seed=84)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    pimax = 50.
    r = SimulationBox2PCF('projected', data1, redges, pimax=pimax, periodic=True, data2=data2)

    # verify with halotools
    pos1 = gather_data(data1, "Position")
    pos2 = gather_data(data2, "Position")
    cf = reference_sim_tpcf(pos1, redges, pimax, data1.attrs['BoxSize'], pos2=pos2)
    assert_allclose(cf, r.corr['corr'])
