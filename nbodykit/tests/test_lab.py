from mpi4py_test import MPIWorld

from nbodykit.lab import *

from nbodykit import setup_logging

# debug logging

setup_logging("debug")

@MPIWorld(NTask=[1, 4])
def test_fftpower(comm):
    cosmo = cosmology.Planck15

    CurrentMPIComm.set(comm)

    # zeldovich particles
    source = Source.ZeldovichParticles(cosmo, nbar=3e-7, redshift=0.55, BoxSize=1380., Nmesh=8, rsd='z', seed=42)

    # compute P(k,mu) and multipoles
    alg = algorithms.FFTPower(source, mode='2d', Nmesh=8, poles=[0,2,4])

    alg.run()

    # and save
    output = "./test_zeldovich.pickle"
    alg.results.save(output)

@MPIWorld(NTask=[2, 3, 4], required=[2, 3, 4])
def test_taskmanager(comm):

    # cosmology
    cosmo = cosmology.Planck15 

    CurrentMPIComm.set(comm)

    cpus_per_task = 2

    with TaskManager(cpus_per_task, debug=True) as tm:

        try:
            for seed in tm.iterate([0, 1, 2]):

                # zeldovich particles
                source = Source.ZeldovichParticles(cosmo, nbar=3e-7, redshift=0.55, BoxSize=1380., Nmesh=8, rsd='z', seed=seed)

                # compute P(k,mu) and multipoles
                alg = algorithms.FFTPower(source, mode='2d', Nmesh=8, poles=[0,2,4])
                alg.run()

                # and save
                output = "./test_batch_zeldovich_seed%d.pickle" % seed
                alg.results.save(output)
        except Exception as e:
            print(e)
            raise
            
