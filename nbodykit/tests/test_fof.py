from mpi4py_test import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_fof(comm):
    cosmo = cosmology.Planck15

    CurrentMPIComm.set(comm)

    # lognormal particles
    source = Source.LogNormal(Plin=cosmology.EHPower(cosmo, 0.55),
                nbar=3e-3, BoxSize=512., Nmesh=128, seed=42)

    # compute P(k,mu) and multipoles
    fof = FOF(source, linking_length=0.2, nmin=20)

    # save the halos
    peaks = fof.find_features()
    peaks.save("FOF-%d" % comm.size, ['Position', 'Velocity', 'Length'])
