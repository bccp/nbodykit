from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

import pytest

# debug logging
setup_logging("debug")


@MPITest([1])
def test_missing_ranks(comm):

    with CurrentMPIComm.enter(comm):
        cpus_per_task = 2
        with pytest.raises(ValueError):
            with TaskManager(cpus_per_task, debug=True, use_all_cpus=True) as tm:
                pass

@MPITest([2])
def test_no_workers(comm):

    with CurrentMPIComm.enter(comm):
        cpus_per_task = 2
        with pytest.raises(ValueError):
            with TaskManager(cpus_per_task, debug=True, use_all_cpus=False) as tm:
                pass


@MPITest([2, 4])
def test_iterate(comm):

    cosmo = cosmology.Planck15

    cpus_per_task = 2
    with CurrentMPIComm.enter(comm):
        with TaskManager(cpus_per_task, debug=True, use_all_cpus=True) as tm:

            try:
                for seed in tm.iterate([0, 1, 2]):

                    # uniform particles
                    source = UniformCatalog(nbar=3e-7, BoxSize=1380., seed=seed)

                    # compute P(k,mu) and multipoles
                    r = FFTPower(source, mode='2d', Nmesh=8, poles=[0,2,4])

                    # and save
                    output = "./test_batch_uniform_seed%d.json" % seed
                    r.save(output)

            except Exception as e:
                print(e)
                raise

@MPITest([4])
def test_map(comm):

    cosmo = cosmology.Planck15

    cpus_per_task = 2

    def fftpower(seed):

        # uniform particles shall not use comm, since this is testing
        # TaskManager setting CurrentMPIComm
        source = UniformCatalog(nbar=3e-7, BoxSize=1380., seed=seed)

        # compute P(k,mu) and multipoles
        r = FFTPower(source, mode='2d', Nmesh=8, poles=[0,2,4])

        # and save
        output = "./test_batch_uniform_seed%d.json" % seed
        r.save(output)

        return seed

    with TaskManager(cpus_per_task, debug=True, use_all_cpus=False, comm=comm) as tm:

        try:
            seeds = [0, 1, 2]
            results = tm.map(fftpower, seeds)
            assert all(seeds[i] == results[i] for i in range(len(seeds)))
        except Exception as e:
            print(e)
            raise
