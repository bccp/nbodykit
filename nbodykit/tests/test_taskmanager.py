from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

# debug logging
setup_logging("debug")

@MPITest([2, 4])
def test_iterate(comm):

    cosmo = cosmology.Planck15 
    CurrentMPIComm.set(comm)

    cpus_per_task = 2
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
            
@MPITest([2, 4])
def test_map(comm):

    cosmo = cosmology.Planck15 
    CurrentMPIComm.set(comm)

    def fftpower(seed):
        
        # uniform particles
        source = UniformCatalog(nbar=3e-7, BoxSize=1380., seed=seed)

        # compute P(k,mu) and multipoles
        r = FFTPower(source, mode='2d', Nmesh=8, poles=[0,2,4])

        # and save
        output = "./test_batch_uniform_seed%d.json" % seed
        r.save(output)
        
        return seed
    
    cpus_per_task = 2
    with TaskManager(cpus_per_task, debug=True, use_all_cpus=True) as tm:

        try:
            seeds = [0, 1, 2]
            results = tm.map(fftpower, seeds)
            assert all(seeds[i] == results[i] for i in range(len(seeds))) 
        except Exception as e:
            print(e)
            raise
            
