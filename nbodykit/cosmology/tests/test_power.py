from nbodykit.cosmology import Cosmology, LinearPower, HalofitPower, ZeldovichPower
from nbodykit.cosmology import EHPower, NoWiggleEHPower
import numpy
from numpy.testing import assert_allclose
import pytest

def test_bad_transfer():

    with pytest.raises(ValueError):
        Plin = LinearPower(Cosmology(), redshift=0., transfer="BAD")

def test_from_astropy():

    from astropy.cosmology import Planck15
    Plin = LinearPower(Planck15, redshift=0)
    Pk = Plin(0.1)

def test_deprecated_ehpower():

    c = Cosmology()
    with pytest.warns(FutureWarning):
        Plin1 = EHPower(c, redshift=0)
        Plin2 = LinearPower(c, 0., transfer='EisensteinHu')
        assert_allclose(Plin1(0.1), Plin2(0.1))

    with pytest.warns(FutureWarning):
        Plin1 = NoWiggleEHPower(c, redshift=0)
        Plin2 = LinearPower(c, 0., transfer='NoWiggleEisensteinHu')
        assert_allclose(Plin1(0.1), Plin2(0.1))

def test_large_scales():

    c = Cosmology()
    k = numpy.logspace(-5, -2, 100)

    # linear power
    Plin = LinearPower(c, redshift=0)

    # nonlinear power
    Pnl = HalofitPower(c, redshift=0)

    # zeldovich power
    Pzel = ZeldovichPower(c, redshift=0)

    assert_allclose(Plin(k), Pnl(k), rtol=1e-2)
    assert_allclose(Plin(k), Pzel(k), rtol=1e-2)

def test_linear_norm():

    # initialize the power
    c = Cosmology().match(sigma8=0.82)
    P = LinearPower(c, redshift=0, transfer='CLASS')

    # compute for array
    k = numpy.logspace(-3, numpy.log10(0.99*c.P_k_max), 100)
    Pk1 = P(k)

    # change sigma8
    P.sigma8 = 0.75
    Pk2 = P(k)
    assert_allclose(Pk1.max()/Pk2.max(), (0.82/0.75)**2, rtol=1e-2)

    # change redshift
    P.redshift = 0.55
    Pk3 = P(k)
    D2 = c.scale_independent_growth_factor(0.)
    D3 = c.scale_independent_growth_factor(0.55)
    assert_allclose(Pk2.max()/Pk3.max(), (D2/D3)**2, rtol=1e-2)

def test_linear():

    # initialize the power
    c = Cosmology().match(sigma8=0.82)
    P = LinearPower(c, redshift=0, transfer='CLASS')

    # check velocity dispersion
    assert_allclose(P.velocity_dispersion(), 5.898, rtol=1e-3)

    # test sigma8
    assert_allclose(P.sigma_r(8.), c.sigma8, rtol=1e-5)

    # change sigma8
    P.sigma8 = 0.80
    c = c.match(sigma8=0.80)

    assert_allclose(P.sigma_r(8.), P.sigma8, rtol=1e-5)

    # change redshift and test sigma8(z)
    P.redshift = 0.55
    assert_allclose(P.sigma_r(8.), c.sigma8_z(P.redshift), rtol=1e-5)

    # desired wavenumbers (in h/Mpc)
    k = numpy.logspace(-3, 2, 500)

    # initialize EH power
    P1 = LinearPower(c, redshift=0., transfer="CLASS")
    P2 = LinearPower(c, redshift=0., transfer='EisensteinHu')
    P3 = LinearPower(c, redshift=0., transfer='NoWiggleEisensteinHu')

    # check different transfers (very roughly)
    Pk1 = P1(k)
    Pk2 = P2(k)
    Pk3 = P3(k)
    assert_allclose(Pk1 / Pk1.max(), Pk2 / Pk2.max(), rtol=0.1)
    assert_allclose(Pk1 / Pk1.max(), Pk3 / Pk3.max(), rtol=0.1)

    # also try scalar
    Pk = P(0.1)

def test_halofit():

    # initialize the power
    c = Cosmology().match(sigma8=0.82)
    P = HalofitPower(c, redshift=0)

    # k is out of range
    with pytest.raises(ValueError):
        Pk = P(2*c.P_k_max)

    # compute for scalar
    Pk = P(0.1)

    # compute for array
    k = numpy.logspace(-3, numpy.log10(0.99*c.P_k_max), 100)
    Pk1 = P(k)

    # change sigma8
    P.sigma8 = 0.75
    Pk2 = P(k)
    assert_allclose(Pk1.max()/Pk2.max(), (0.82/0.75)**2, rtol=1e-2)

    # change redshift
    P.redshift = 0.55
    Pk3 = P(k)
    D2 = c.scale_independent_growth_factor(0.)
    D3 = c.scale_independent_growth_factor(0.55)
    assert_allclose(Pk2.max()/Pk3.max(), (D2/D3)**2, rtol=1e-2)

def test_zeldovich():

    # initialize the power
    c = Cosmology().match(sigma8=0.82)
    P = ZeldovichPower(c, redshift=0, nmax=1)

    # compute for scalar
    Pk = P(0.1)

    # compute for array
    k = numpy.logspace(-3, numpy.log10(0.99*c.P_k_max), 100)
    Pk1 = P(k)

    # change sigma8
    P.sigma8 = 0.75
    Pk2 = P(k)
    assert_allclose(Pk1.max()/Pk2.max(), (0.82/0.75)**2, rtol=1e-2)

    # change redshift
    P.redshift = 0.55
    Pk3 = P(k)
    D2 = c.scale_independent_growth_factor(0.)
    D3 = c.scale_independent_growth_factor(0.55)
    assert_allclose(Pk2.max()/Pk3.max(), (D2/D3)**2, rtol=1e-2)
