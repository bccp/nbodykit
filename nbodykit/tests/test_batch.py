from mpi4py import MPI
from nbodykit import setup_logging
from nbodykit import CurrentMPIComm
from nbodykit.batch import TaskManager
from nbodykit.source.catalog import UniformCatalog
from nbodykit.algorithms import FFTPower

import pytest

# debug logging
setup_logging("debug")


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi(min_size=2)
def test_iterate(comm):

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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi(min_size=2)
def test_map(comm):

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
