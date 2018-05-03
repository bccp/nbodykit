from nbodykit.lab import *
from nbodykit import setup_logging
import os

setup_logging()
data_dir = os.path.split(os.path.abspath(__file__))[0]

def gather_data(source, name):
    return numpy.concatenate(source.comm.allgather(source[name].compute()), axis=0)

def generate_sim_data_1d(seed):
    return UniformCatalog(nbar=3e-4, BoxSize=512., seed=seed)

def generate_sim_data_angular(seed):
    s = UniformCatalog(nbar=1000, BoxSize=1.0, seed=seed)
    s['RA'], s['DEC'] = transform.CartesianToEquatorial(s['Position'], observer=0.5*s.attrs['BoxSize'])
    return s

def generate_survey_data(seed):

    # make the data
    cosmo = cosmology.Planck15
    d = UniformCatalog(nbar=3e-4, BoxSize=512., seed=seed)
    d['RA'], d['DEC'], d['Redshift'] = transform.CartesianToSky(d['Position'], cosmo)

    # make the randoms (ensure nbar is high enough to not have missing values)
    r = UniformCatalog(nbar=3e-4, BoxSize=512., seed=seed*2)
    r['RA'], r['DEC'], r['Redshift'] = transform.CartesianToSky(r['Position'], cosmo)

    return d, r

@CurrentMPIComm.enable
def test_1d_sim_nonperiodic_auto(comm=None):

    # uniform source of particles
    source = generate_sim_data_1d(seed=42)
    randoms = generate_sim_data_1d(seed=84)

    # make the bin edges
    redges = numpy.linspace(0.01, 20, 10)

    # compute 2PCF
    r = SimulationBox2PCF('1d', source, redges, periodic=False, randoms1=randoms)

    pos_d,pos_r = gather_data(source,'Position'), gather_data(randoms,'Position')
    # save
    if comm.rank == 0:
        numpy.savetxt(os.path.join(data_dir,'test_1d_sim_data.dat'),pos_d)
        numpy.savetxt(os.path.join(data_dir,'test_1d_sim_randoms.dat'),pos_r)
        numpy.savetxt(os.path.join(data_dir,'test_1d_sim_nonperiodic_auto.dat'),r.corr['corr'])

@CurrentMPIComm.enable
def test_1d_survey_auto(comm=None):
    cosmo = cosmology.Planck15

    # data and randoms
    data, randoms = generate_survey_data(seed=42)

    # make the bin edges
    redges = numpy.linspace(1.0, 10, 5)

    # compute 2PCF
    r = SurveyData2PCF('1d', data, randoms, redges, cosmo=cosmo)

    # save
    rdz_d = numpy.vstack([gather_data(data,key) for key in ['RA','DEC','Redshift']]).T
    rdz_r = numpy.vstack([gather_data(randoms,key) for key in ['RA','DEC','Redshift']]).T
    if comm.rank == 0:
        numpy.savetxt(os.path.join(data_dir,'test_1d_survey_data.dat'),rdz_d)
        numpy.savetxt(os.path.join(data_dir,'test_1d_survey_randoms.dat'),rdz_r)
        numpy.savetxt(os.path.join(data_dir,'test_1d_survey_auto.dat'),numpy.vstack([r.D1D2['npairs'],r.D1R2['npairs'],r.D2R1['npairs'],r.R1R2['npairs'],r.corr['corr']]).T)

@CurrentMPIComm.enable
def test_angular_sim_nonperiodic_auto(comm=None):

    # uniform source of particles
    source = generate_sim_data_angular(seed=42)
    randoms = generate_sim_data_angular(seed=84)

    # make the bin edges
    theta_edges = numpy.linspace(0.1, 10.0, 20)

    # compute 2PCF
    r = SimulationBox2PCF('angular', source, theta_edges, periodic=False, randoms1=randoms)

    # save
    pos_d,pos_r = gather_data(source,'Position'), gather_data(randoms,'Position')
    if comm.rank == 0:
        numpy.savetxt(os.path.join(data_dir,'test_angular_sim_data.dat'),pos_d)
        numpy.savetxt(os.path.join(data_dir,'test_angular_sim_randoms.dat'),pos_r)
        numpy.savetxt(os.path.join(data_dir,'test_angular_sim_nonperiodic_auto.dat'),r.corr['corr'])

@CurrentMPIComm.enable
def test_angular_survey_auto(comm):

    # uniform source of particles
    data = generate_sim_data_angular(seed=42)
    randoms = generate_sim_data_angular(seed=84)

    # make the bin edges
    theta_edges = numpy.linspace(0.1, 10.0, 20)

    # compute 2PCF
    r = SurveyData2PCF('angular', data, randoms, theta_edges)

    # save
    rd_d = numpy.vstack([gather_data(data,key) for key in ['RA','DEC']]).T
    rd_r = numpy.vstack([gather_data(randoms,key) for key in ['RA','DEC']]).T
    if comm.rank == 0:
        numpy.savetxt(os.path.join(data_dir,'test_angular_survey_data.dat'),rd_d)
        numpy.savetxt(os.path.join(data_dir,'test_angular_survey_randoms.dat'),rd_r)
        numpy.savetxt(os.path.join(data_dir,'test_angular_survey_auto.dat'),numpy.vstack([r.D1D2['npairs'],r.D1R2['npairs'],r.D2R1['npairs'],r.R1R2['npairs'],r.corr['corr']]).T)


test_1d_sim_nonperiodic_auto()
test_1d_survey_auto()
test_angular_sim_nonperiodic_auto()
test_angular_survey_auto()
