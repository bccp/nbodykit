from nbodykit.cosmology import Cosmology
from nbodykit.cosmology import EHPower
from nbodykit.cosmology import NoWiggleEHPower
from nbodykit.cosmology import Planck15
from nbodykit.cosmology import Cosmology
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

def test_eprime():
    import autograd
    C = Cosmology(Om0=0.3, Ode0=0.7, Neff=0, Tcmb=0)
    print C.Onu0, C.Ogamma0, C.Ode0, C.Om0, C.Ok0
    a = numpy.linspace(0.01, 1.0, 20, endpoint=True)

    def E(a):
        return C.efunc(1 / a - 1)
    def Eprime(a):
        return C.efunc_prime(1 / a - 1)

    def E2(a):
        a = numpy.array(a)
        return (C.Om0 *a ** -3 + (C.Ogamma0 + C.Onu0) * a ** -4 + C.Ode0 + C.Ok0 * a ** -2 )**0.5 

    E2prime = autograd.elementwise_grad(E2)

    for aa in a:
        err = check_grad(E, Eprime, x0=numpy.array([aa]), epsilon=1e-7)
        print err / E(aa), E(aa), E2(aa), Eprime(aa), E2prime(aa)

def test_ode():

    C = Cosmology(Om0=0.3, Ode0=0.7, Neff=0, Tcmb0=0)
    assert C.Ogamma0 == 0

    a = numpy.linspace(0.1, 1.0, 11, endpoint=True)
    z = 1 / a - 1

    D1, f1, D2, f2 = C.lptode(z)
    D1 /= D1[-1]
#    assert_allclose(f1, C.Om(z=z) ** (5./9))
    assert_allclose(f1, C.growth_rate(z), rtol=1e-6)
    assert_allclose(D1, C.growth_function(z), rtol=1e-6)


