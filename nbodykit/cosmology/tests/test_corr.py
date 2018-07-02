from nbodykit.cosmology import CorrelationFunction
from nbodykit.cosmology import Cosmology, xi_to_pk, pk_to_xi
from nbodykit.cosmology import LinearPower, HalofitPower, ZeldovichPower
import numpy
from numpy.testing import assert_allclose

def test_mcfit():

    c = Cosmology()
    Plin = LinearPower(c, redshift=0, transfer='EisensteinHu')

    for ell in [0, 2, 4]:
        # do Pk to CF; use Plin for ell>0 just for testing
        k = numpy.logspace(-4, 2, 1024)
        CF = pk_to_xi(k, Plin(k), ell=ell)

        # do CF to Pk; use Plin for ell>0 just for testing
        r = numpy.logspace(-2, 4, 1024)
        Pk2 = xi_to_pk(r, CF(r), ell=ell)(k)

        idx = (k>1e-2)&(k<10.)
        assert_allclose(Pk2[idx], Plin(k[idx]), rtol=1e-2)

def test_linear():

    # linear power
    c = Cosmology()
    Plin = LinearPower(c, redshift=0, transfer='EisensteinHu')

    # desired separation (in Mpc/h)
    r = numpy.logspace(0, numpy.log10(150), 500)

    # linear correlation
    CF = CorrelationFunction(Plin)
    CF.sigma8 = 0.82
    xi1 = CF(100.)

    assert_allclose(CF.redshift, CF.attrs['redshift'])
    assert_allclose(CF.sigma8, CF.attrs['sigma8'])

    # change sigma8
    CF.sigma8 = 0.75
    xi2 = CF(100.)
    assert_allclose(xi1/xi2, (0.82/0.75)**2, rtol=1e-2)

    # change redshift
    CF.redshift = 0.55
    xi3 = CF(100.)
    D2 = CF.cosmo.scale_independent_growth_factor(0.)
    D3 = c.scale_independent_growth_factor(0.55)
    assert_allclose(xi2.max()/xi3.max(), (D2/D3)**2, rtol=1e-2)



def test_halofit():

    # nonlinear power
    Pnl = HalofitPower(Cosmology(), redshift=0)

    # desired separation (in Mpc/h)
    r = numpy.logspace(0, numpy.log10(150), 500)

    # nonlinear correlation
    CF = CorrelationFunction(Pnl)

    xi = CF(r)

def test_zeldovich():

    # zeldovich power
    Pzel = ZeldovichPower(Cosmology(), redshift=0, nmax=1, transfer='EisensteinHu')

    # desired separation (in Mpc/h)
    r = numpy.logspace(0, numpy.log10(150), 500)

    # zeldovich correlation
    CF = CorrelationFunction(Pzel)

    xi = CF(r)
