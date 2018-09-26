from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from scipy.interpolate import InterpolatedUnivariateSpline
from numpy.testing import assert_allclose, assert_array_equal
import pytest

setup_logging("debug")

NDATA = 1000
NBAR = 1e-4

def make_sources(cosmo, comm):

    data = RandomCatalog(NDATA, seed=42, comm=comm)
    randoms = RandomCatalog(NDATA*10, seed=84, comm=comm)

    # add the random columns
    for s in [data, randoms]:

        # ra, dec, z
        s['z']   = s.rng.normal(loc=0.5, scale=0.1)
        s['ra']  = s.rng.uniform(low=110, high=260)
        s['dec'] = s.rng.uniform(low=-3.6, high=60.)

        # position
        s['Position'] = transform.SkyToCartesian(s['ra'], s['dec'], s['z'], cosmo=cosmo)

    return data, randoms

@MPITest([1])
def test_add_fkpweight(comm):
    cosmo = cosmology.Planck15

    # make the sources
    data, randoms = make_sources(cosmo, comm)
    for s in [data, randoms]:
        # constant number density
        s['NZ'] = NBAR

    P0 = 1e4

    # two mesh objects from same FKP source
    fkp = FKPCatalog(data, randoms, P0=P0)

    assert_allclose(fkp['data']['FKPWeight'].compute(),
                    FKPWeightFromNbar(P0, fkp['data']['NZ'].compute()))

    assert_allclose(fkp['data']['FKPWeight'].compute(),
                    FKPWeightFromNbar(P0, fkp['data']['NZ'].compute()))

    # updating NZ shall affect FKPWeight if it is the default.
    for s in [data, randoms]:
        # constant number density
        s['NZ'] = 2 * NBAR

    assert_allclose(fkp['data']['FKPWeight'].compute(),
                    FKPWeightFromNbar(P0, fkp['data']['NZ'].compute()))

    assert_allclose(fkp['data']['FKPWeight'].compute(),
                    FKPWeightFromNbar(P0, fkp['data']['NZ'].compute()))

@MPITest([1, 4])
def test_diff_cross_boxsizes(comm):

    cosmo = cosmology.Planck15
    P0 = 1e4

    # make the sources
    data, randoms = make_sources(cosmo, comm)
    for s in [data, randoms]:

        # constant number density
        s['NZ'] = NBAR

        # completeness weights
        s['Weight'] = (1 + P0*s['NZ'])**2


    # two mesh objects from same FKP source
    fkp = FKPCatalog(data, randoms, P0=P0)
    mesh1 = fkp.to_mesh(Nmesh=128, dtype='f8')

    # second mesh has larger box size
    mesh2 = fkp.to_mesh(Nmesh=128, dtype='f8', BoxSize=1.1*mesh1.attrs['BoxSize'])

    # compute the multipoles
    r = ConvolvedFFTPower(mesh1, second=mesh2, poles=[0,2,4], dk=0.005)

    # make sure we matched the box sizes
    assert_array_equal(r.first.attrs['BoxSize'], r.second.attrs['BoxSize'])
    assert_array_equal(r.first.attrs['BoxCenter'], r.second.attrs['BoxCenter'])

@MPITest([1, 4])
def test_true_cross_corr_fail(comm):

    cosmo = cosmology.Planck15
    P0 = 1e4

    # make the sources
    data, randoms = make_sources(cosmo, comm)
    for s in [data, randoms]:
        s['NZ'] = NBAR

    # two mesh objects from different FKP source
    fkp1 = FKPCatalog(data, randoms, P0=P0)
    fkp2 = FKPCatalog(data, randoms, P0=P0)

    mesh1 = fkp1.to_mesh(Nmesh=128, dtype='f8')
    mesh2 = fkp2.to_mesh(Nmesh=128, dtype='f8')

    # cannot do cross correlations with different data/randoms catalogs
    with pytest.raises(NotImplementedError):
        r = ConvolvedFFTPower(mesh1, second=mesh2, poles=[0,2,4], dk=0.005)


@MPITest([1, 4])
def test_bad_cross_corr_columns(comm):

    cosmo = cosmology.Planck15

    # make the sources
    data, randoms = make_sources(cosmo, comm)

    for s in [data, randoms]:
        s['NZ'] = NBAR

    # same FKP source but different selection columns won't work!
    fkp = FKPCatalog(data, randoms)
    fkp['data']['Selection2'] = fkp['data/Selection']
    fkp['randoms']['Selection2'] = fkp['randoms/Selection']

    mesh1 = fkp.to_mesh(Nmesh=128, dtype='f8', selection='Selection')
    mesh2 = fkp.to_mesh(Nmesh=128, dtype='f8', selection='Selection2')

    # columns in mesh must be same except for weight
    # YF: FIXME: why this limitation?
    with pytest.raises(NotImplementedError):
        r = ConvolvedFFTPower(mesh1, second=mesh2, poles=[0,2,4], dk=0.005)

@MPITest([1, 4])
def test_cross_corr(comm):

    cosmo = cosmology.Planck15

    # make the sources
    data, randoms = make_sources(cosmo, comm)
    P0 = 1e4

    for s in [data, randoms]:

        # constant number density
        s['NZ'] = NBAR

        # completeness weights
        s['Weight'] = (1 + P0*s['NZ'])**2

    # two mesh objects from same FKP source
    fkp = FKPCatalog(data, randoms, P0=P0)
    mesh1 = fkp.to_mesh(Nmesh=128, dtype='f8', comp_weight='Weight', selection='Selection')
    mesh2 = fkp.to_mesh(Nmesh=128, dtype='f8', comp_weight='Weight', selection='Selection')

    # compute the multipoles
    r = ConvolvedFFTPower(mesh1, second=mesh2, poles=[0,2,4], dk=0.005)

    # normalization
    assert_allclose(r.attrs['data.norm'], NDATA*NBAR)
    assert_allclose(r.attrs['randoms.norm'], NDATA*NBAR)

    # shotnoise
    S_data = r.attrs['first.data.W']/r.attrs['randoms.norm']
    S_ran = r.attrs['first.randoms.W']/r.attrs['randoms.norm']*r.attrs['alpha']**2
    S = S_data + S_ran
    assert_allclose(S, r.attrs['shotnoise'])

@MPITest([1, 4])
def test_bad_input(comm):

    cosmo = cosmology.Planck15

    # source has wrong type
    data, randoms = make_sources(cosmo, comm)
    for s in [data, randoms]:
        s['NZ'] = NBAR
        s['FKPWeight'] = 1.0 / (1 + 2e4*s['NZ'])

    with pytest.raises(TypeError):
        r = ConvolvedFFTPower(data, poles=[0,2,4], dk=0.005, Nmesh=64)

    # the FKP source
    fkp = FKPCatalog(data, randoms)

    for s in [data, randoms]:
        assert_allclose(s['FKPWeight'], 1.0 / (1 + 2e4*s['NZ']))

    # must specify P0_FKP
    with pytest.raises(ValueError):
        r = ConvolvedFFTPower(fkp, poles=0, dk=0.005, use_fkp_weights=True, P0_FKP=None, Nmesh=64)

    # warn about overwriting FKP Weights
    with pytest.raises(ValueError):
        r = ConvolvedFFTPower(fkp, poles=0, dk=0.005, use_fkp_weights=True, P0_FKP=1e4, Nmesh=64)


@MPITest([4])
def test_no_monopole(comm):

    cosmo = cosmology.Planck15

    # make the sources
    data, randoms = make_sources(cosmo, comm)

    # select in given redshift range
    for s in [data, randoms]:
        s['NZ'] = NBAR

    # the FKP source
    fkp = FKPCatalog(data, randoms, nbar='NZ')
    fkp = fkp.to_mesh(Nmesh=128, dtype='f8')

    # compute the multipoles
    r = ConvolvedFFTPower(fkp, poles=[2,4], dk=0.005)

    assert 'power_0' not in r.poles.variables
    assert 'power_2' in r.poles.variables
    assert 'power_4' in r.poles.variables

@MPITest([4])
def test_bad_normalization(comm):

    cosmo = cosmology.Planck15

    # make the sources
    data, randoms = make_sources(cosmo, comm)

    # select in given redshift range
    for s in [data, randoms]:
        s['NZ'] = NBAR

    # bad normalization via NZ
    randoms['NZ'] *= 50.0

    # the FKP source
    fkp = FKPCatalog(data, randoms, nbar='NZ')
    fkp = fkp.to_mesh(Nmesh=128, dtype='f8')

    # compute the multipoles
    with pytest.raises(ValueError):
        r = ConvolvedFFTPower(fkp, poles=[0,2,4], dk=0.005)

@MPITest([4])
def test_selection(comm):

    cosmo = cosmology.Planck15

    # make the sources
    data, randoms = make_sources(cosmo, comm)

    # select in given redshift range
    for s in [data, randoms]:
        s['NZ'] = NBAR
        s['Selection'] = (s['z'] > 0.4)&(s['z'] < 0.6)

    # the FKP source
    fkp = FKPCatalog(data, randoms, nbar='NZ')
    fkp = fkp.to_mesh(Nmesh=128, dtype='f8', selection='Selection')

    # compute the multipoles
    r = ConvolvedFFTPower(fkp, poles=[0,2,4], dk=0.005)

    # number of data objects selected
    N = comm.allreduce(((data['z'] > 0.4)&(data['z'] < 0.6)).sum())
    assert_allclose(r.attrs['data.N'], N)

    # number of randoms selected
    N = comm.allreduce(((randoms['z'] > 0.4)&(randoms['z'] < 0.6)).sum())
    assert_allclose(r.attrs['randoms.N'], N)

    # and save
    r.save("conv-power-with-selection.json")

    # load and check output
    r2 = ConvolvedFFTPower.load("conv-power-with-selection.json", comm=comm)
    assert_array_equal(r.poles.data, r2.poles.data)

@MPITest([1, 4])
def test_run(comm):

    cosmo = cosmology.Planck15
    P0 = 1e4

    # make the sources
    data, randoms = make_sources(cosmo, comm)
    for s in [data, randoms]:

        # constant number density
        s['NZ'] = NBAR

        # completeness weights
        s['Weight'] = (1 + P0*s['NZ'])**2

    # the FKP source
    fkp = FKPCatalog(data, randoms, P0=P0, nbar='NZ')
    fkp = fkp.to_mesh(Nmesh=128, dtype='f8', fkp_weight='FKPWeight', comp_weight='Weight', selection='Selection')

    # compute the multipoles
    r = ConvolvedFFTPower(fkp, poles=[0,2,4], dk=0.005)

    # compute pkmu
    mu_edges = numpy.linspace(0, 1, 6)
    pkmu = r.to_pkmu(mu_edges=mu_edges, max_ell=4)

    # normalization
    assert_allclose(r.attrs['data.norm'], NDATA*NBAR)
    assert_allclose(r.attrs['randoms.norm'], NDATA*NBAR)

    # shotnoise
    S_data = r.attrs['data.W']/r.attrs['randoms.norm']
    S_ran = r.attrs['randoms.W']/r.attrs['randoms.norm']*r.attrs['alpha']**2
    S = S_data + S_ran
    assert_allclose(S, r.attrs['shotnoise'])

@MPITest([1, 4])
def test_run_unique_bins(comm):

    cosmo = cosmology.Planck15
    P0 = 1e4

    # make the sources
    data, randoms = make_sources(cosmo, comm)
    for s in [data, randoms]:

        # constant number density
        s['NZ'] = NBAR

        # completeness weights
        s['Weight'] = (1 + P0*s['NZ'])**2

    # the FKP source
    fkp = FKPCatalog(data, randoms, P0=P0, nbar='NZ')
    fkp = fkp.to_mesh(Nmesh=128, dtype='f8', fkp_weight='FKPWeight', comp_weight='Weight', selection='Selection')

    # compute the multipoles
    r = ConvolvedFFTPower(fkp, poles=[0,2,4], dk=0)

    # compute pkmu
    mu_edges = numpy.linspace(0, 1, 6)
    pkmu = r.to_pkmu(mu_edges=mu_edges, max_ell=4)
    assert_allclose(pkmu.coords['k'], r.poles.coords['k'])

@MPITest([1, 4])
def test_run_unique_bins_windowonly(comm):

    cosmo = cosmology.Planck15
    P0 = 1e4

    # make the sources
    data, randoms = make_sources(cosmo, comm)
    for s in [data, randoms]:

        # constant number density
        s['NZ'] = NBAR

        # completeness weights
        s['Weight'] = (1 + P0*s['NZ'])**2

    # the FKP source
    fkp = FKPCatalog(data=randoms, randoms=None, P0=P0, nbar='NZ')
    fkp = fkp.to_mesh(Nmesh=128, dtype='f8', fkp_weight='FKPWeight', comp_weight='Weight', selection='Selection')

    # compute the multipoles
    r = ConvolvedFFTPower(fkp, poles=[0,2,4], dk=0)

    # compute pkmu
    mu_edges = numpy.linspace(0, 1, 6)
    pkmu = r.to_pkmu(mu_edges=mu_edges, max_ell=4)
    assert_allclose(pkmu.coords['k'], r.poles.coords['k'])

@MPITest([1, 4])
def test_window_only(comm):
    NDATA = 1000
    NBAR = 1e-4
    FSKY = 0.15
    P0 = 1e4

    cosmo = cosmology.Planck15

    # make the sources
    data, randoms = make_sources(cosmo, comm)

    for s in [data, randoms]:

        # constant number density
        s['NZ'] = NBAR

        # completeness weights
        s['Weight'] = (1 + P0*s['NZ'])**2

    empty = randoms[:0]
    # initialize the FKP source with random as data, for measuring the window.
    fkp = FKPCatalog(data=randoms, randoms=None, P0=P0)

    # compute the multipoles
    r = ConvolvedFFTPower(fkp.to_mesh(Nmesh=128), poles=[0,2,4], dk=0.005)

    assert not numpy.isnan(r.poles['power_0']).any()
    assert not numpy.isnan(r.poles['power_2']).any()
    assert not numpy.isnan(r.poles['power_4']).any()

@MPITest([1, 4])
def test_with_zhist(comm):

    NDATA = 1000
    NBAR = 1e-4
    FSKY = 0.15

    cosmo = cosmology.Planck15

    # make the sources
    data, randoms = make_sources(cosmo, comm)

    # compute NZ from randoms
    zhist = RedshiftHistogram(randoms, FSKY, cosmo, redshift='z')

    # normalize NZ to the total size of the data catalog
    alpha = 1.0 * data.csize / randoms.csize
    # add n(z) from randoms to the FKP source

    randoms['NZ'] = zhist.interpolate(randoms['z']) * alpha
    data['NZ'] = zhist.interpolate(data['z']) * alpha

    # initialize the FKP source
    fkp = FKPCatalog(data, randoms)

    # compute the multipoles
    r = ConvolvedFFTPower(fkp.to_mesh(Nmesh=128), poles=[0,2,4], dk=0.005)

    assert_allclose(r.attrs['data.norm'], 0.000388338522187, rtol=1e-5)
    assert_allclose(r.attrs['randoms.norm'], 0.000395808747269, rtol=1e-5)
