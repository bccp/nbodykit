from nbodykit.cosmology import Cosmology
from nbodykit.cosmology import EHPower
from nbodykit.cosmology import NoWiggleEHPower
from nbodykit.cosmology import Planck15
from nbodykit.cosmology import Cosmology
from nbodykit.cosmology import PerturbationGrowth

from numpy.testing import assert_allclose, assert_array_equal
import pytest
import numpy

def test_read_only():
    with pytest.raises(ValueError):
        Planck15['Om0'] = 0.31

def test_missing():

    cosmo = Cosmology(sigma8=0.9)
    sigma8 = cosmo['sigma8']
    assert sigma8 == 0.9

    Ok0 = cosmo['Ok0']
    assert Ok0 == cosmo.engine.Ok0

    with pytest.raises(KeyError):
        invalid = cosmo['missing_param']

def test_wCDM():

    cosmo = Cosmology(w0=-0.9, flat=True)
    w0 = cosmo.w0
    assert w0 == -0.9

    from astropy.cosmology import FlatwCDM
    assert isinstance(cosmo.engine, FlatwCDM)

def test_growth_rate():
    a = numpy.linspace(0.1, 1.0, 20, endpoint=True)
    z = 1 / a - 1
    f = Planck15.growth_rate(z=z)
    assert_allclose(f, Planck15.Om(z=z) ** (5./9), rtol=1e-2)

def test_fast():
    a = numpy.linspace(0.1, 1.0, 20, endpoint=True)
    z = 1 / a - 1
    f1 = Planck15.growth_rate(z=z)

    fit = Planck15.growth_rate.fit('z', range=(0, 100), bins=1000)
    f2 = fit(z)

    assert_allclose(f1, f2, rtol=1e-4)

def test_efunc_prime():
    epsilon = 1e-4
    z = numpy.linspace(0, 3, 100.0)

    # cosmology with no massive neutrinos
    cosmo = Cosmology()
    d1 = cosmo.efunc_prime(z)
    d2 = (cosmo.efunc(z+epsilon) - cosmo.efunc(z-epsilon))/(2*epsilon) * -(1+z)**2
    assert_allclose(d1, d2, err_msg="efunc_prime error for cosmo with no massive neutrinos", rtol=1e-3)

    # cosmology with massive neutrinos
    cosmo = Planck15
    d1 = cosmo.efunc_prime(z)
    d2 = (cosmo.efunc(z+epsilon) - cosmo.efunc(z-epsilon))/(2*epsilon) * -(1+z)**2
    assert_allclose(d1, d2, err_msg="efunc_prime error for cosmo with massive neutrinos", rtol=1e-3)

def test_clone_m_nu():
    c1 = Planck15.clone(m_nu=[0, 0, 0])

    # the new parameter
    assert_allclose(c1.m_nu, 0)

    # the old parameters are the same
    for name in c1:
        if name in Planck15 and name != 'm_nu':
            assert_array_equal(c1[name], Planck15[name])


def test_clone_Om():
    c1 = Planck15.clone(Om0=0.8)

    # the new parameter
    assert_allclose(c1.Om0, 0.8)

    # the old parameters are the same
    for name in c1:
        if name in Planck15 and name != 'Om0':
            assert_array_equal(c1[name], Planck15[name])

def test_clone_sigma8():
    c1 = Planck15.clone(sigma8=0.9)

    # the new parameter
    assert_allclose(c1.sigma8, 0.9)

    # the old parameters are the same
    for name in c1:
        if name in Planck15 and name != 'sigma8':
            assert_array_equal(c1[name], Planck15[name])

try:
    from classylss.binding import ClassEngine, Background
except ImportError:
    ClassEngine = None
    pass

@pytest.mark.skipif(ClassEngine is None, reason='class binding is not installed')
def test_ode():
    C = Planck15

    CC = ClassEngine.from_astropy(C.engine)
    bg = Background(CC)

    a = numpy.logspace(-2, 0, 11, endpoint=True)

    pt = PerturbationGrowth(C, a=a)

    z = 1 / a - 1

    f1 = pt.f1(a)
    D2 = pt.D2(a)
    f2 = pt.f2(a)

    """
    from scipy.optimize import minimize

    def f(a_H):
        pt.a_H = a_H
        pt._D1, pt._D2 = pt._solve()
        D1 = pt.D1(a)
        D_CC = bg.scale_independent_growth_factor(z)
        return ((D1 / D_CC - 1) ** 2).sum() ** 0.5
    a_Hbest = minimize(f, pt.a_H, method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-7)).x

    print(a_Hbest)
    pt.a_H = a_Hbest
    """

    D1 = pt.D1(a)
    D_CC = bg.scale_independent_growth_factor(z)
    assert_allclose(D1, D_CC, rtol=1e-4)
