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


    # zeldovich particles
    source = Source.ZeldovichParticles(Plin=cosmology.EHPower(cosmo, redshift),
                                        nbar=3e-3, BoxSize=BoxSize, Nmesh=128, rsd=[0, 0, 1], seed=42)

    
    # run FOF
    fof = FOF(source, linking_length=0.2, nmin=20)
    halos = HaloFinder(source, fof['HaloLabel'])

    # FIXME: HaloFinder is only nonzero on root
    s = comm.bcast(halos._source)
    halos = Source.Array(s)
    
    # convert to a HaloCatalog
    # FIXME: the position/velocity units are wrong!
    cat = Source.HaloCatalog(halos, particle_mass=1e12, cosmo=cosmo, redshift=redshift, mdef='vir')
    cat['Selection'] = cat['HaloMass'] > 0
    
    # make the HOD catalog
    hod = Source.HOD(cat.to_halotools(BoxSize), cosmo=cosmo, redshift=redshift, mdef='vir')
    
    # and repopulate in-place once
    hod.repopulate()