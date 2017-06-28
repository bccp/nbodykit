from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_periodic_box_autocorr(comm):

    import kdcount.correlate as correlate
    numpy.random.seed(42)
    CurrentMPIComm.set(comm)

    # a log-normal source
    cosmo = cosmology.Planck15
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift=0.55),
                              nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)

    # add some weights b/w 0 and 1
    source['Weight'] = numpy.random.random(size=len(source))

    # make the bin edges
    edges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    r = SimulationBoxPairCount(source, edges, periodic=True, weight='Weight')

    pos = numpy.concatenate(comm.allgather(source['Position'].compute()), axis=0)
    w = numpy.concatenate(comm.allgather(source['Weight'].compute()), axis=0)

    # use kdcount to verify
    tree = correlate.points(pos, boxsize=source.attrs['BoxSize'], weights=w)
    bins = correlate.RBinning(edges)
    pc = correlate.paircount(tree, tree, bins, np=0, usefast=False, compute_mean_coords=True)

    # verify mean coordinates
    assert_allclose(pc.mean_centers, r.result['r'])

    # verify num pairs
    assert_allclose(pc.pair_counts, r.result['npairs'])

    # verify weighted pair count
    assert_allclose(pc.sum1, r.result['npairs'] * r.result['weightavg'])


@MPITest([1, 4])
def test_nonperiodic_box_autocorr(comm):

    import kdcount.correlate as correlate
    numpy.random.seed(42)
    CurrentMPIComm.set(comm)

    # a log-normal source
    cosmo = cosmology.Planck15
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift=0.55),
                              nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)

    # add some weights b/w 0 and 1
    source['Weight'] = numpy.random.random(size=len(source))

    # make the bin edges
    edges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    r = SimulationBoxPairCount(source, edges, periodic=False, weight='Weight')

    pos = numpy.concatenate(comm.allgather(source['Position'].compute()), axis=0)
    w = numpy.concatenate(comm.allgather(source['Weight'].compute()), axis=0)

    # use kdcount to verify
    tree = correlate.points(pos, weights=w)
    bins = correlate.RBinning(edges)
    pc = correlate.paircount(tree, tree, bins, np=0, usefast=False, compute_mean_coords=True)

    # verify mean coordinates
    assert_allclose(pc.mean_centers, r.result['r'], rtol=1e-5)

    # verify num pairs
    assert_allclose(pc.pair_counts, r.result['npairs'])

    # verify weighted pair count
    assert_allclose(pc.sum1, r.result['npairs'] * r.result['weightavg'], rtol=1e-5)


@MPITest([1, 4])
def test_periodic_box_crosscorr(comm):

    import kdcount.correlate as correlate
    numpy.random.seed(42)
    CurrentMPIComm.set(comm)

    # a log-normal source
    cosmo = cosmology.Planck15
    source1 = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift=0.55),
                              nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)
    source2 = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift=0.),
                              nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)

    # make the bin edges
    edges = numpy.linspace(10, 150, 10)

    # do the weighted paircount
    r = SimulationBoxPairCount(source1, edges, source2=source2, periodic=True)

    pos1 = numpy.concatenate(comm.allgather(source1['Position'].compute()), axis=0)
    pos2 = numpy.concatenate(comm.allgather(source2['Position'].compute()), axis=0)

    # use kdcount to verify
    tree1 = correlate.points(pos1, boxsize=source1.attrs['BoxSize'])
    tree2 = correlate.points(pos2, boxsize=source2.attrs['BoxSize'])
    bins = correlate.RBinning(edges)
    pc = correlate.paircount(tree1, tree2, bins, np=0, usefast=False, compute_mean_coords=True)

    # verify mean coordinates
    assert_allclose(pc.mean_centers, r.result['r'])

    # verify num pairs
    assert_allclose(pc.pair_counts, r.result['npairs'])

    # verify weighted pair count
    assert_allclose(pc.sum1, r.result['npairs'] * r.result['weightavg'])
