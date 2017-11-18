from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose, assert_array_equal
import os

setup_logging("debug")
data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')

@MPITest([4])
def test_sim_threeptcf(comm):

    import tempfile

    CurrentMPIComm.set(comm)
    BoxSize = 400.0

    # load the test data
    filename = os.path.join(data_dir, 'threeptcf_sim_data.dat')
    cat = CSVCatalog(filename, names=['x', 'y', 'z', 'w'])
    cat['Position'] = transform.StackColumns(cat['x'], cat['y'], cat['z'])
    cat['Position'] *= BoxSize

    # r binning
    nbins = 8
    edges = numpy.linspace(0, 200.0, nbins+1)

    # run the algorithm
    ells = list(range(0, 11))
    r = SimulationBox3PCF(cat, ells, edges, BoxSize=BoxSize, weight='w')

    # load the result from file
    truth = numpy.empty((8,8,11))
    with open(os.path.join(data_dir, 'threeptcf_sim_result.dat'), 'r') as ff:
        for line in ff:
            fields = line.split()
            i, j = int(fields[0]), int(fields[1])
            truth[i,j,:] = list(map(float, fields[2:]))
            truth[j,i,:] = truth[i,j,:]

    # test equality
    for i, ell in enumerate(ells):
        x = r.poles['corr_%d' %ell]
        assert_allclose(x * (4*numpy.pi)**2 / (2*ell+1), truth[...,i], rtol=1e-3, err_msg='mismatch for ell=%d' %ell)

    # save to temp file
    filename = 'test-threept-cf.json'
    r.save(filename)
    r2 = SimulationBox3PCF.load(filename)
    assert_array_equal(r.poles.data, r2.poles.data)

    if comm.rank == 0:
        os.remove(filename)


@MPITest([4])
def test_survey_threeptcf(comm):

    import tempfile

    CurrentMPIComm.set(comm)
    BoxSize = 400.0
    cosmo = cosmology.Planck15

    # load the test data
    filename = os.path.join(data_dir, 'threeptcf_sim_data.dat')
    cat = CSVCatalog(filename, names=['x', 'y', 'z', 'w'])
    cat['Position'] = transform.StackColumns(cat['x'], cat['y'], cat['z'])
    cat['Position'] *= BoxSize

    # place observer at center of box
    cat['Position'] -= 0.5*BoxSize

    # transform to RA/DEC/Z
    # NOTE: we update Position here to get exact same Position as SurveyData3PCF
    cat['RA'], cat['DEC'], cat['Z'] = transform.CartesianToSky(cat['Position'], cosmo)
    cat['Position'] = transform.SkyToCartesian(cat['RA'], cat['DEC'], cat['Z'], cosmo)

    # r binning
    nbins = 8
    edges = numpy.linspace(0, 200.0, nbins+1)

    # run the reference (non-periodic simulation box)
    ells = list(range(0, 11))
    ref = SimulationBox3PCF(cat, ells, edges, BoxSize=BoxSize, weight='w', periodic=False)

    # run the algorithm to test
    r = SurveyData3PCF(cat, ells, edges, cosmo, weight='w', ra='RA', dec='DEC', redshift='Z')

    # test equality between survey result and reference
    for i, ell in enumerate(ells):
        a = r.poles['corr_%d' %ell]
        b = ref.poles['corr_%d' %ell]
        assert_allclose(a, b, rtol=1e-5, err_msg='mismatch for ell=%d' %ell)

    # save to temp file
    filename = 'test-threept-cf.json'
    r.save(filename)
    r2 = SurveyData3PCF.load(filename)
    assert_array_equal(r.poles.data, r2.poles.data)

    if comm.rank == 0:
        os.remove(filename)
