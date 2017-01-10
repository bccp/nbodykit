from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_allclose

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_bianchi(comm):
    
    NDATA = 1000
    NBAR = 1e-4
    
    CurrentMPIComm.set(comm)
    cosmo = cosmology.Planck15
    
    data = Source.RandomParticles(NDATA, seed=42)
    randoms = Source.RandomParticles(NDATA*10, seed=84)
    
    # add the random columns
    for s in [data, randoms]:
        
        # ra, dec, z
        s['z']   = s.rng.normal(loc=0.5, scale=0.1, size=s.size)
        s['ra']  = s.rng.uniform(low=110, high=260, size=s.size)
        s['dec'] = s.rng.uniform(low=-3.6, high=60., size=s.size)
        
        # position
        s['Position'] = transform.SkyToCartesion(s['ra'], s['dec'], s['z'], cosmo=cosmo)
    
        # constant number density
        s['NZ'] = NBAR
        
        # FKP weights
        P0 = 1e4
        s['FKPWeight'] = 1.0 / (1 + P0*s['NZ'])
        
        # completeness weights
        s['Weight'] = 1.0/s['FKPWeight']**2
    
    # the FKP source
    fkp = Source.FKPCatalog(data, randoms)
    fkp = fkp.to_mesh(Nmesh=128, dtype='f8', nbar='NZ', fkp_weight='FKPWeight', comp_weight='Weight')

    # compute the multipoles
    alg = BianchiFFTPower(fkp, max_ell=4, dk=0.005)
    alg.run()

    # tests
    assert_allclose(alg.attrs['data.A'], NDATA*NBAR)
    assert_allclose(alg.attrs['randoms.A'], NDATA*NBAR)


    


