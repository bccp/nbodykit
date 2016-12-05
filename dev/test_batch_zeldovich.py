from nbodykit.lab import *

# debug logging
setup_logging(logging.DEBUG)

# global comm and rank
global_comm = MPI.COMM_WORLD
rank = global_comm.rank

# cosmology
cosmo = cosmology.default_cosmology.get()

cpus_per_task = 2
with TaskManager(cpus_per_task, debug=True) as tm:

    comm = tm.comm    
    for seed in tm.iterate([0, 1, 2]):
        
        # zeldovich particles
        source = Source.ZeldovichParticles(comm, cosmo, nbar=3e-4, redshift=0.55, BoxSize=1380., Nmesh=256, rsd='z', seed=seed)

        # compute P(k,mu) and multipoles
        alg = algorithms.FFTPower(comm, source, mode='2d', Nmesh=256, poles=[0,2,4])
        edges, pkmu, poles = alg.run()

        # and save
        output = "./test_batch_zeldovich_seed%d.pickle" %seed
        result = alg.save(output, edges=edges, pkmu=pkmu, poles=poles)
