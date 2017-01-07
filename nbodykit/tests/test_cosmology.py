from nbodykit.cosmology import Cosmology
from nbodykit.cosmology import EHPower
from nbodykit.cosmology import NoWiggleEHPower
from nbodykit.cosmology import Planck15
from nbodykit.cosmology import Cosmology
from nbodykit.cosmology import PerturbationGrowth

from numpy.testing import assert_allclose
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

def test_ode():
    C = Cosmology(Om0=0.3, Ode0=0.7, Neff=0, Tcmb0=0)
    assert C.Ogamma0 == 0
    pt = PerturbationGrowth(C)

    a = numpy.linspace(0.1, 1.0, 11, endpoint=True)
    z = 1 / a - 1
    D1 = pt.D1(a)
    f1 = pt.f1(a)
    D2 = pt.D2(a)
    f2 = pt.f2(a)

    assert_allclose(D1, C.growth_function(z), rtol=1e-6)

    assert_allclose(f1, C.growth_rate(z), rtol=1e-6)


