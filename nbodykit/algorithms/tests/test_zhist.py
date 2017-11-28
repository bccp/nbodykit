from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose, assert_array_equal

# debug logging
setup_logging("debug")

@MPITest([1])
def test_save(comm):

    N = 1000
    FSKY = 1.0
    
    CurrentMPIComm.set(comm)
    cosmo = cosmology.Planck15
    
    # create the source
    source = RandomCatalog(N, seed=42)
    source['z'] = source.rng.normal(loc=0.5, scale=0.1, size=source.size)
    
    # compute the histogram
    r = RedshiftHistogram(source, FSKY, cosmo, redshift='z')
    r.run()    
    r.save('zhist-test.json')

    r2 = RedshiftHistogram.load('zhist-test.json')

    assert_array_equal(r.bin_edges, r.bin_edges)
    assert_array_equal(r.bin_centers, r2.bin_centers)
    assert_array_equal(r.dV, r2.dV)
    assert_array_equal(r.nbar, r2.nbar)
    for k in r.attrs:
        assert_array_equal(r.attrs[k], r2.attrs[k])

@MPITest([1, 4])
def test_unweighted(comm):
    
    N = 1000
    FSKY = 1.0
    
    CurrentMPIComm.set(comm)
    cosmo = cosmology.Planck15
    
    # create the source
    source = RandomCatalog(N, seed=42)
    source['z'] = source.rng.normal(loc=0.5, scale=0.1, size=source.size)
    
    # compute the histogram
    r = RedshiftHistogram(source, FSKY, cosmo, redshift='z')

    assert (r.nbar*r.dV).sum() == N
    

@MPITest([1, 4])
def test_weighted(comm):
    
    N = 1000
    FSKY = 1.0
    
    CurrentMPIComm.set(comm)
    cosmo = cosmology.Planck15
    
    # create the source
    source = RandomCatalog(N, seed=42)
    source['z'] = source.rng.normal(loc=0.5, scale=0.1, size=source.size)
    source['weight'] = source.rng.uniform(0, high=1., size=source.size)
    
    # compute the histogram
    r = RedshiftHistogram(source, FSKY, cosmo, redshift='z', weight='weight')

    wsum = comm.allreduce(source['weight'].sum().compute())
    assert_allclose((r.nbar*r.dV).sum(), wsum)




    


