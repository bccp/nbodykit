from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_array_equal, assert_allclose
import kdcount.correlate as correlate
import os
import pytest
from mpi4py import MPI

setup_logging()

def make_corrfunc_input(data, cosmo):
    ra = gather_data(data, 'RA')
    dec = gather_data(data, 'DEC')
    z = gather_data(data, 'Redshift')
    rdist = cosmo.comoving_distance(z)
    return numpy.vstack([ra, dec, rdist]).T

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_sim_data(seed, comm):
    return UniformCatalog(nbar=3e-4, BoxSize=512., seed=seed, comm=comm)

def generate_survey_data(seed, comm):

    # make the data
    cosmo = cosmology.Planck15
    d = UniformCatalog(nbar=3e-4, BoxSize=512., seed=seed, comm=comm)
    d['RA'], d['DEC'], d['Redshift'] = transform.CartesianToSky(d['Position'], cosmo)

    # make the randoms (ensure nbar is high enough to not have missing values)
    r = UniformCatalog(nbar=3e-4, BoxSize=512., seed=seed*2, comm=comm)
    r['RA'], r['DEC'], r['Redshift'] = transform.CartesianToSky(r['Position'], cosmo)

    return d, r

def reference_sim_tpcf(pos1, redges, Nmu, BoxSize, randoms=None, pos2=None):
    """Refernce 2D 2PCF using halotools"""
    from halotools.mock_observables import s_mu_tpcf

    mu_bins = numpy.linspace(0, 1, Nmu+1)
    estimator = 'Natural' if randoms is None else 'Landy-Szalay'
    do_auto = True if pos2 is None else False
    return s_mu_tpcf(pos1, redges, mu_bins, period=BoxSize, sample2=pos2,
                    randoms=randoms, estimator=estimator, do_auto=do_auto)

def reference_survey_tpcf(data1, randoms1, redges, Nmu, data2=None, randoms2=None):
    """Reference 2D 2PCF using Corrfunc"""
    from Corrfunc.utils import convert_3d_counts_to_cf
    from Corrfunc.mocks import DDsmu_mocks

    if data2 is None:
        data2 = data1
    if randoms2 is None:
        randoms2 = randoms1

    # updack the columns to pass to Corrfunc
    ra_d1, dec_d1, r_d1 = data1.T
    ra_r1, dec_r1, r_r1 = randoms1.T
    ra_d2, dec_d2, r_d2 = data2.T
    ra_r2, dec_r2, r_r2 = randoms2.T

    # the Corrfunc keywords
    kws = {}
    kws['cosmology'] = 1
    kws['nthreads'] = 1
    kws['nmu_bins'] = Nmu
    kws['mu_max'] = 1.0
    kws['binfile'] = redges
    kws['is_comoving_dist'] = True

    # do the pair counting
    D1D2 = DDsmu_mocks(0, RA1=ra_d1, DEC1=dec_d1, CZ1=r_d1, RA2=ra_d2, DEC2=dec_d2, CZ2=r_d2, **kws)
    D1R2 = DDsmu_mocks(0, RA1=ra_d1, DEC1=dec_d1, CZ1=r_d1, RA2=ra_r2, DEC2=dec_r2, CZ2=r_r2, **kws)
    D2R1 = DDsmu_mocks(0, RA1=ra_d2, DEC1=dec_d2, CZ1=r_d2, RA2=ra_r1, DEC2=dec_r1, CZ2=r_r1, **kws)
    R1R2 = DDsmu_mocks(0, RA1=ra_r1, DEC1=dec_r1, CZ1=r_r1, RA2=ra_r2, DEC2=dec_r2, CZ2=r_r2, **kws)

    # combine using Landy-Szalay
    ND1 = len(ra_d1); ND2 = len(ra_d2)
    NR1 = len(ra_r1); NR2 = len(ra_r2)
    CF = convert_3d_counts_to_cf(ND1, ND2, NR1, NR2, D1D2, D1R2, D2R1, R1R2).reshape((-1,Nmu))

    D1D2 = D1D2.reshape((-1,Nmu))
    D1R2 = D1R2.reshape((-1,Nmu))
    D2R1 = D2R1.reshape((-1,Nmu))
    R1R2 = R1R2.reshape((-1,Nmu))
    return D1D2, D1R2, D2R1, R1R2, CF

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sim_periodic_auto(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    Nmu = 10
    r = SimulationBox2PCF('2d', source, redges, Nmu=Nmu, periodic=True)

    # verify with halotools
    pos = gather_data(source, "Position")
    cf = reference_sim_tpcf(pos, redges, Nmu, source.attrs['BoxSize'])
    assert_allclose(cf, r.corr['corr'])

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sim_nonperiodic_auto(comm):

    # uniform source of particles
    source = generate_sim_data(seed=42, comm=comm)
    randoms = generate_sim_data(seed=84, comm=comm)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    Nmu = 10
    r = SimulationBox2PCF('2d', source, redges, periodic=False, Nmu=Nmu, randoms1=randoms)

    # verify with halotools
    pos_d = gather_data(source, "Position")
    pos_r = gather_data(randoms, "Position")
    cf = reference_sim_tpcf(pos_d, redges, Nmu, None, randoms=pos_r)
    assert_allclose(cf, r.corr['corr'], rtol=1e-5, atol=1e-3)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sim_periodic_cross(comm):

    # uniform source of particles
    data1 = generate_sim_data(seed=42, comm=comm)
    data2 = generate_sim_data(seed=84, comm=comm)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    Nmu = 10
    r = SimulationBox2PCF('2d', data1, redges, Nmu=Nmu, periodic=True, data2=data2)

    # verify with halotools
    pos1 = gather_data(data1, "Position")
    pos2 = gather_data(data2, "Position")
    cf = reference_sim_tpcf(pos1, redges, Nmu, data1.attrs['BoxSize'], pos2=pos2)
    assert_allclose(cf, r.corr['corr'])

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_survey_cross(comm):
    cosmo = cosmology.Planck15

    # data and randoms
    data1, randoms1 = generate_survey_data(seed=42, comm=comm)
    data2, randoms2 = generate_survey_data(seed=84, comm=comm)

    # make the bin edges
    redges = numpy.linspace(0.01, 10, 5)
    Nmu = 3

    # compute 2PCF
    r = SurveyData2PCF('2d', data1, randoms1, redges, Nmu=Nmu, cosmo=cosmo, data2=data2, randoms2=randoms2)

    # run Corrfunc to verify
    data1 = make_corrfunc_input(data1, cosmo)
    randoms1 = make_corrfunc_input(randoms1, cosmo)
    data2 = make_corrfunc_input(data2, cosmo)
    randoms2 = make_corrfunc_input(randoms2, cosmo)
    D1D2, D1R2, D2R1, R1R2, cf = reference_survey_tpcf(data1, randoms1, redges, Nmu,
                                                        data2=data2, randoms2=randoms2)

    # verify pair counts and CF
    assert_allclose(cf, r.corr['corr'])
    assert_allclose(D1D2['npairs'], r.D1D2['npairs'])
    assert_allclose(D1R2['npairs'], r.D1R2['npairs'])
    assert_allclose(D2R1['npairs'], r.D2R1['npairs'])
    assert_allclose(R1R2['npairs'], r.R1R2['npairs'])


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_xil(comm):
    cosmo = cosmology.Planck15

    # data and randoms
    data1, randoms1 = generate_survey_data(seed=42, comm=comm)
    data2, randoms2 = generate_survey_data(seed=84, comm=comm)

    # make the bin edges
    redges = numpy.linspace(0.01, 10, 5)
    Nmu = 10
    poles = [0,2,4]

    # compute 2PCF
    r = SurveyData2PCF('2d', data1, randoms2, redges, Nmu=Nmu, cosmo=cosmo, data2=data2, randoms2=randoms2)
    xil = r.corr.to_poles(poles)
    xil = r.corr.sel(mu=slice(0.2, 0.5), method='nearest').to_poles(poles)

