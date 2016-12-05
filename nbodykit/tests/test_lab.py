from mpi4py_test import MPIWorld
from nbodykit.lab import *

@MPIWorld(NTask=[1, 4])
def test_fftpower(comm):
    cosmo = cosmology.default_cosmology.get()

    # debug logging
    setup_logging(logging.DEBUG)

    # zeldovich particles
    source = Source.ZeldovichParticles(comm, cosmo, nbar=3e-4, redshift=0.55, BoxSize=1380., Nmesh=256, rsd='z', seed=42)

    # compute P(k,mu) and multipoles
    alg = algorithms.FFTPower(comm, source, mode='2d', Nmesh=256, poles=[0,2,4])
    edges, pkmu, poles = alg.run()

    # and save
    output = "./test_zeldovich.pickle"
    result = alg.save(output, edges=edges, pkmu=pkmu, poles=poles)



