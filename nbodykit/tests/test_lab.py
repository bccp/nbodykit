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
    source = Source.ZeldovichParticles(cosmo, nbar=3e-4, redshift=0.55, BoxSize=1380., Nmesh=8, rsd='z', seed=42)

    # compute P(k,mu) and multipoles
    alg = algorithms.FFTPower(source, mode='2d', Nmesh=8, poles=[0,2,4])
    edges, pkmu, poles = alg.run()

    # and save
    output = "./test_zeldovich.pickle"
    result = alg.save(output, edges=edges, pkmu=pkmu, poles=poles)

@MPIWorld(NTask=[3], required=[3])
def test_taskmanager(comm):

    # cosmology
    cosmo = cosmology.Planck15 

    CurrentMPIComm.set(comm)

    cpus_per_task = 2

    with TaskManager(cpus_per_task, debug=True, comm=comm) as tm:

        for seed in tm.iterate([0, 1, 2]):

            # zeldovich particles
            source = Source.ZeldovichParticles(cosmo, nbar=3e-4, redshift=0.55, BoxSize=1380., Nmesh=8, rsd='z', seed=seed, comm=tm.comm)

            # compute P(k,mu) and multipoles
            alg = algorithms.FFTPower(source, mode='2d', Nmesh=8, poles=[0,2,4], comm=tm.comm)
            edges, pkmu, poles = alg.run()

            # and save
            output = "./test_batch_zeldovich_seed%d.pickle" %seed
            result = alg.save(output, edges=edges, pkmu=pkmu, poles=poles)

