from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_fof(comm):
    cosmo = cosmology.Planck15

    CurrentMPIComm.set(comm)

    # lognormal particles
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=512., Nmesh=128, seed=42)

    # compute P(k,mu) and multipoles
    fof = FOF(source, linking_length=0.2, nmin=20)
    source['Density'] = KDDensity(source, margin=1).density

    # save the halos
    peaks = fof.find_features(peakcolumn='Density')
    peaks.save("FOF-%d" % comm.size, ['CMPosition', 'CMVelocity', 'Length', 'PeakPosition', 'PeakVelocity'])
