from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_fftpower(comm):
    cosmo = cosmology.Planck15

    CurrentMPIComm.set(comm)

    # lognormal particles
    source = Source.LogNormal(Plin=cosmology.EHPower(cosmo, redshift=0.55),
                    nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)
                    
    # apply RSD
    source['Position'] += source['VelocityOffset'] * [0,0,1]

    # compute P(k,mu) and multipoles
    result = FFTPower(source, mode='2d', poles=[0,2,4], los=[0,0,1])

    # and save
    output = "./test_zeldovich-%d.json" % comm.size
    result.save(output)

@MPITest([1, 4])
def test_paint(comm):
    cosmo = cosmology.Planck15

    CurrentMPIComm.set(comm)

    # lognormal particles
    source = Source.LogNormal(Plin=cosmology.EHPower(cosmo, redshift=0.55),
                    nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)
                    
    # apply RSD
    source['Position'] += source['VelocityOffset'] * [0,0,1]

    # convert to mesh, with Painter specifics
    source = source.to_mesh(Nmesh=64, BoxSize=1380., interlaced=True, window='tsc', compensated=True)

    def filter(k, v):
        kk = sum(ki ** 2 for ki in k)
        kk[kk == 0] = 1
        return v / kk

    source.apply(filter)

    real = source.paint(mode='real')
    complex = source.paint(mode='complex')

    real.save(output="./test_paint-real-%d.bigfile" % comm.size)
    complex.save(output="./test_paint-complex-%d.bigfile" % comm.size)

