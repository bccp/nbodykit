from nbodykit.lab import *
from nbodykit import setup_logging
from mpi4py_test import MPITest
from numpy.testing import assert_allclose

import dask
dask.set_options(get=dask.get)

@MPITest([1, 4])
def test_lpt(comm):
    cosmo = cosmology.Planck15

    linear = Source.LinearMesh(Plin=cosmology.EHPower(cosmo, 0.0),
                BoxSize=1280, Nmesh=64, seed=42, comm=comm)

    lpt = Source.LPTParticles(dlink=linear.to_field(mode='complex'), cosmo=cosmo, redshift=0.0)

    # let's compsre the large scale growth against linear theory.
    Dsquare = []
    redshifts = [9, 5, 2, 1, 0]
    for z in redshifts:
        lpt.set_redshift(z)
        r = FFTPower(lpt, Nmesh=64, mode='1d')
        use = r.power['k'] < 0.02
        pbar = r.power['power'][use].mean()
        Dsquare.append(pbar)

    # linear theory and za at this scale can have 1 few percent of difference.
    assert_allclose(Dsquare, lpt.cosmo.growth_function(redshifts) ** 2 * Dsquare[-1], rtol=1e-2)

    v2 = comm.allreduce((lpt['LPTDisp1']**2).sum(axis=0).compute())
    vdisp = ((v2 / lpt.csize) ** 0.5)

    # should be almost isotropic. Need to add a term in linear to fix the modulus.
    assert_allclose(vdisp, vdisp.mean(), rtol=5e-2)

@MPITest([1])
def test_lpt_grad(comm):
    cosmo = cosmology.Planck15

    linear = Source.LinearMesh(Plin=cosmology.EHPower(cosmo, 0.0),
                BoxSize=1280, Nmesh=8, seed=42, comm=comm)

    dlink = linear.to_field(mode='complex')
    lpt0 = Source.LPTParticles(dlink=dlink, cosmo=cosmo, redshift=0.0)

    def chi2(lpt):
        return (lpt['Position'] ** 2).sum().compute()

    def grad_chi2(lpt):
        lpt['GradPosition'] = 2 * lpt['Position']
        return lpt.gradient()

    chi2_0 = chi2(lpt0)
    grad_a = grad_chi2(lpt0)

    mode = (1, 2, 3)
    dlink1 = dlink.copy()
    diff = 0.003
    dlink1.real[mode] += diff
    lpt1 = Source.LPTParticles(dlink=dlink1, cosmo=cosmo, redshift=0.0)

    chi2_1 = chi2(lpt1)

    grad_n = (chi2_1 - chi2_0) / diff

    print(grad_n, grad_a[mode])
