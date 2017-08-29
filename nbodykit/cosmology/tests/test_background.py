from nbodykit.cosmology import Planck15, Cosmology, PerturbationGrowth
from numpy.testing import assert_allclose
import numpy

def test_ode():

    C = Planck15
    a = numpy.logspace(-2, 0, 11, endpoint=True)
    z = 1 / a - 1

    pt = PerturbationGrowth(C, a=a)

    """
    from scipy.optimize import minimize
    def f(a_H):
        pt.a_H = a_H
        pt._D1, pt._D2 = pt._solve()
        D1 = pt.D1(a)
        D_CC = C.scale_independent_growth_factor(z)
        return ((D1 / D_CC - 1) ** 2).sum() ** 0.5
    a_Hbest = minimize(f, pt.a_H, method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-7)).x
    print(a_Hbest)
    pt.a_H = a_Hbest
    """
    # linear growth function
    D1 = pt.D1(a)
    D_CC = C.scale_independent_growth_factor(z)
    assert_allclose(D1, D_CC, rtol=1e-3)

    # linear growth rate
    f1 = pt.f1(a)
    f_CC = C.scale_independent_growth_rate(z)
    assert_allclose(f1, f_CC, rtol=1e-3)

    # second order quantities
    D2 = pt.D2(a)
    f2 = pt.f2(a)
