from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_unweighted(comm):
    
    N = 1000
    FSKY = 1.0
    
    CurrentMPIComm.set(comm)
    cosmo = cosmology.Planck15
    
    # create the source
    source = Source.RandomParticles(N, seed=42)
    source['z'] = source.rng.normal(loc=0.5, scale=0.1, size=source.size)
    
    # compute the histogram
    alg = RedshiftHistogram(source, FSKY, cosmo, redshift='z')
    alg.run()

    assert (alg.nbar*alg.dV).sum() == N
    

@MPITest([1, 4])
def test_weighted(comm):
    
    N = 1000
    FSKY = 1.0
    
    CurrentMPIComm.set(comm)
    cosmo = cosmology.Planck15
    
    # create the source
    source = Source.RandomParticles(N, seed=42)
    source['z'] = source.rng.normal(loc=0.5, scale=0.1, size=source.size)
    source['weight'] = source.rng.uniform(0, high=1., size=source.size)
    
    # compute the histogram
    alg = RedshiftHistogram(source, FSKY, cosmo, redshift='z', weight='weight')
    alg.run()

    assert_allclose((alg.nbar*alg.dV).sum(), (source['weight'].sum()).compute())




    


