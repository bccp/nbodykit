from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose

setup_logging()

@MPITest([1,4])
def test_paint(comm):

    cosmo = cosmology.Planck15

    # linear grid
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LinearMesh(Plin, Nmesh=64, BoxSize=512, seed=42, comm=comm)

    # compute P(k) from linear grid
    r = FFTPower(source, mode='1d', Nmesh=64, dk=0.01, kmin=0.005)

    # run and get the result
    valid = r.power['modes'] > 0

    # variance of each point is 2*P^2/N_modes
    theory = Plin(r.power['k'][valid])
    errs = (2*theory**2/r.power['modes'][valid])**0.5

    # compute reduced chi-squared of measurement to theory
    chisq = ((r.power['power'][valid].real - theory)/errs)**2
    N = valid.sum()
    red_chisq = chisq.sum() / (N-1)

    # make sure it is less than 1.5 (should be ~1)
    assert red_chisq < 1.5, "reduced chi sq of linear grid measurement = %.3f" %red_chisq
