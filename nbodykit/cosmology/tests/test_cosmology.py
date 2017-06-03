from nbodykit.cosmology import Cosmology
from nbodykit.cosmology import EHPower
from nbodykit.cosmology import NoWiggleEHPower
from nbodykit.cosmology import Planck15
from nbodykit.cosmology import Cosmology
from nbodykit.cosmology import PerturbationGrowth

from numpy.testing import assert_allclose
import pytest
import numpy
from scipy.optimize import check_grad

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


