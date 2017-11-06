from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_array_equal, assert_allclose
import kdcount.correlate as correlate
import os
import pytest

setup_logging()

def make_corrfunc_input(data, cosmo):
    ra = gather_data(data, 'RA')
    dec = gather_data(data, 'DEC')
    z = gather_data(data, 'Redshift')
    rdist = cosmo.comoving_distance(z)
    return numpy.vstack([ra, dec, rdist]).T

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_sim_data(seed):
    return UniformCatalog(nbar=3e-4, BoxSize=512., seed=seed)

def generate_survey_data(seed):

    # make the data
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    d = LogNormalCatalog(Plin=Plin, nbar=3e-7, BoxSize=1380, Nmesh=32, seed=seed)
    d['RA'], d['DEC'], d['Redshift'] = transform.CartesianToSky(d['Position'], cosmo)

    # make the randoms (ensure nbar is high enough to not have missing values)
    r = UniformCatalog(nbar=3e-7, BoxSize=1380., seed=seed*2)
    r['RA'], r['DEC'], r['Redshift'] = transform.CartesianToSky(r['Position'], cosmo)

    return d, r

def reference_sim_tpcf(pos1, redges, BoxSize, randoms=None, pos2=None):
    """Reference 1D 2PCF using halotools"""
    from halotools.mock_observables import tpcf

    estimator = 'Natural' if randoms is None else 'Landy-Szalay'
    do_auto = True if pos2 is None else False
    return tpcf(pos1, redges, period=BoxSize, sample2=pos2, randoms=randoms,
                estimator=estimator, do_auto=do_auto)

def reference_survey_tpcf(pos1, pos2, redges):
    """Reference 1D 2PCF using Corrfunc"""
    from Corrfunc.utils import convert_3d_counts_to_cf
    from Corrfunc.mocks import DDsmu_mocks

    ra1, dec1, r1 = pos1.T
    ra2, dec2, r2 = pos2.T

    kws = {}
    kws['cosmology'] = 1
    kws['nthreads'] = 1
    kws['nmu_bins'] = 1
    kws['mu_max'] = 1.0
    kws['binfile'] = redges
    kws['is_comoving_dist'] = True
    kws['output_savg'] = True

    DD = DDsmu_mocks(1, RA1=ra1, DEC1=dec1, CZ1=r1, **kws)
    DR = DDsmu_mocks(0, RA1=ra1, DEC1=dec1, CZ1=r1, RA2=ra2, DEC2=dec2, CZ2=r2, **kws)
    RR = DDsmu_mocks(1, RA1=ra2, DEC1=dec2, CZ1=r2, **kws)

    N1 = len(ra1); N2 = len(ra2)
    return DD, DR, RR, convert_3d_counts_to_cf(N1, N1, N2, N2, DD, DR, DR, RR)

@MPITest([1, 3])
def test_sim_periodic_auto(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    source = generate_sim_data(seed=42)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    r = SimulationBox2PCF('1d', source, redges, periodic=True)

    # verify with halotools
    pos = gather_data(source, "Position")
    cf = reference_sim_tpcf(pos, redges, source.attrs['BoxSize'])
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
    r = SimulationBox2PCF('1d', source, redges, periodic=False, randoms1=randoms)

    # verify with halotools
    pos_d = gather_data(source, "Position")
    pos_r = gather_data(randoms, "Position")
    cf = reference_sim_tpcf(pos_d, redges, None, randoms=pos_r)
    assert_allclose(cf, r.corr['corr'], rtol=1e-5, atol=1e-5)

@MPITest([1, 3])
def test_sim_periodic_cross(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    data1 = generate_sim_data(seed=42)
    data2 = generate_sim_data(seed=84)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    r = SimulationBox2PCF('1d', data1, redges, periodic=True, data2=data2)

    # verify with halotools
    pos1 = gather_data(data1, "Position")
    pos2 = gather_data(data2, "Position")
    cf = reference_sim_tpcf(pos1, redges, data1.attrs['BoxSize'], pos2=pos2)
    assert_allclose(cf, r.corr['corr'])

@MPITest([1, 3])
def test_survey_auto(comm):
    cosmo = cosmology.Planck15
    CurrentMPIComm.set(comm)

    # data and randoms
    data, randoms = generate_survey_data(seed=42)

    # make the bin edges
    redges = numpy.linspace(1.0, 50, 5)

    # compute 2PCF
    r = SurveyData2PCF('1d', data, randoms, redges, cosmo=cosmo)

    # run Corrfunc to verify
    pos1 = make_corrfunc_input(data, cosmo)
    pos2 = make_corrfunc_input(randoms, cosmo)
    DD, DR, RR, cf = reference_survey_tpcf(pos1, pos2, redges)

    # verify pair counts and CF
    assert_allclose(cf, r.corr['corr'])
    assert_allclose(DD['npairs'], r.D1D2['npairs'])
    assert_allclose(DR['npairs'], r.D1R2['npairs'])
    assert_allclose(RR['npairs'], r.R1R2['npairs'])

@MPITest([1])
def test_low_nbar_randoms(comm):
    CurrentMPIComm.set(comm)

    # uniform source of particles
    source = generate_sim_data(seed=42)
    randoms = UniformCatalog(nbar=3e-6, BoxSize=512., seed=84)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 50)

    # compute 2PCF
    with pytest.warns(UserWarning):
        r = SimulationBox2PCF('1d', source, redges, periodic=False, randoms1=randoms)
