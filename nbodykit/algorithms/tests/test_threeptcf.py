from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose, assert_array_equal
import os

setup_logging("debug")

# The test result data (threeptcf_sim_result.dat) is computed with
# Daniel Eisenstein's
# C++ implementation on the same input data set for poles up to l=11;
# We shall agree with it to high precision.
#
# If we need to reproduced these files:
# Nick Hand sent the code and instructions to Yu Feng on Aug-20-2018.

data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')

@MPITest([4])
def test_sim_threeptcf(comm):

    import tempfile

    BoxSize = 400.0

    # load the test data
    filename = os.path.join(data_dir, 'threeptcf_sim_data.dat')
    cat = CSVCatalog(filename, names=['x', 'y', 'z', 'w'], comm=comm)
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
    r2 = SimulationBox3PCF.load(filename, comm=comm)
    assert_array_equal(r.poles.data, r2.poles.data)

    if comm.rank == 0:
        os.remove(filename)


@MPITest([4])
def test_survey_threeptcf(comm):

    import tempfile

    BoxSize = 400.0
    cosmo = cosmology.Planck15

    # load the test data
    filename = os.path.join(data_dir, 'threeptcf_sim_data.dat')
    cat = CSVCatalog(filename, names=['x', 'y', 'z', 'w'], comm=comm)
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
    r2 = SurveyData3PCF.load(filename, comm=comm)
    assert_array_equal(r.poles.data, r2.poles.data)

    if comm.rank == 0:
        os.remove(filename)

@MPITest([1])
def test_sim_threeptcf_pedantic(comm):

    BoxSize = 400.0

    # load the test data
    filename = os.path.join(data_dir, 'threeptcf_sim_data.dat')
    cat = CSVCatalog(filename, names=['x', 'y', 'z', 'w'], comm=comm)
    cat['Position'] = transform.StackColumns(cat['x'], cat['y'], cat['z'])
    cat['Position'] *= BoxSize

    cat = cat[::20]

    # r binning
    nbins = 8
    edges = numpy.linspace(0, 200.0, nbins+1)

    # run the algorithm
    ells = [0, 2, 4, 8]
    r = SimulationBox3PCF(cat, ells, edges, BoxSize=BoxSize, weight='w')
    p_fast = r.run()
    p_pedantic = r.run(pedantic=True)

    # test equality
    for i, ell in enumerate(sorted(ells)):
        x1 = p_fast['corr_%d' %ell]
        x2 = p_pedantic['corr_%d' %ell]
        assert_allclose(x1, x2)

@MPITest([1])
def test_sim_threeptcf_shuffled(comm):

    BoxSize = 400.0

    # load the test data
    filename = os.path.join(data_dir, 'threeptcf_sim_data.dat')
    cat = CSVCatalog(filename, names=['x', 'y', 'z', 'w'], comm=comm)
    cat['Position'] = transform.StackColumns(cat['x'], cat['y'], cat['z'])
    cat['Position'] *= BoxSize

    cat = cat

    # r binning
    nbins = 8
    edges = numpy.linspace(0, 200.0, nbins+1)

    # run the algorithm
    ells = list(range(0, 2))[::-1]
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
    for i, ell in enumerate(sorted(ells)):
        x = r.poles['corr_%d' %ell]
        assert_allclose(x * (4*numpy.pi)**2 / (2*ell+1), truth[...,i], rtol=1e-3, err_msg='mismatch for ell=%d' %ell)
