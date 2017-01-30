from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

setup_logging()

@MPITest([4])
def test_hod(comm):
    
    CurrentMPIComm.set(comm)
    
    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # lognormal particles
    source = Source.LogNormal(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42)
    
    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir')
        
    # make the HOD catalog from halotools catalog
    hod = Source.HOD(halos.to_halotools(), seed=42)
        
    # RSD offset in 'z' direction
    hod['Position'] += hod['VelocityOffset'] * [0, 0, 1]
    
    # compute the power
    r = FFTPower(hod.to_mesh(Nmesh=128), mode='2d', Nmu=5, los=[0,0,1])
    