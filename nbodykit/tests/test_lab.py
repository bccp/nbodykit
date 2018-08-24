from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging, set_options, GlobalCache

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_fftpower(comm):
    cosmo = cosmology.Planck15

    with CurrentMPIComm.enter(comm):
        # lognormal particles
        Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
        source = LogNormalCatalog(Plin=Plin, nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)

        # apply RSD
        source['Position'] += source['VelocityOffset'] * [0,0,1]

        # compute P(k,mu) and multipoles
        result = FFTPower(source, mode='2d', poles=[0,2,4], los=[0,0,1])

        # and save
        output = "./test_fftpower-%d.json" % comm.size
        result.save(output)

@MPITest([1, 4])
def test_compute(comm):
    cosmo = cosmology.Planck15

    with CurrentMPIComm.enter(comm):
        # lognormal particles
        Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
        source = LogNormalCatalog(Plin=Plin, nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)

        # apply RSD
        source['Position'] += source['VelocityOffset'] * [0,0,1]

        # convert to mesh, with Painter specifics
        source = source.to_mesh(Nmesh=64, BoxSize=1380., interlaced=True, window='tsc', compensated=True)

        def filter(k, v):
            kk = sum(ki ** 2 for ki in k)
            kk[kk == 0] = 1
            return v / kk

        source = source.apply(filter)

        real = source.compute(mode='real')
        complex = source.compute(mode='complex')

        source.save(output="./test_paint-real-%d.bigfile" % comm.size, mode='real')
        source.save(output="./test_paint-complex-%d.bigfile" % comm.size, mode='complex')

@MPITest([1, 4])
def test_current_mpicomm(comm):
    cosmo = cosmology.Planck15

    with CurrentMPIComm.enter(comm):
        pass

@MPITest([1, 4])
def test_set_options(comm):

    with CurrentMPIComm.enter(comm):
        with set_options(global_cache_size=5e9, dask_chunk_size=75):
            s = UniformCatalog(1000, 1.0)

            # check cache size
            cache = GlobalCache.get()
            assert cache.cache.available_bytes == 5e9

            # check chunk size
            assert s['Position'].chunks[0][0] == 75

        s = UniformCatalog(1000, 1.0)
        assert s['Position'].chunks[0][0] == s.size
