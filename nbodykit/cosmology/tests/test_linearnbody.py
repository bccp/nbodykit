from nbodykit.cosmology.linearnbody import LinearNbody
from numpy.testing import assert_allclose

def test_linear_nbody():
    from nbodykit.cosmology import Cosmology
    cosmo = Cosmology(Omega0_cdm = 0.26, Omega0_b = 0.04,
                      m_ncdm=[0.6], P_k_max=1e-2)

    linearnbody = LinearNbody(cosmo, c_b = 0, c_ncdm_1ev_z0=0)

    a0 = 0.01

    k, q0, p0 = linearnbody.seed_from_synchronous(cosmo, a0)

    k, qt, pt = linearnbody.seed_from_synchronous(cosmo, 1.0)

    a, q, p = linearnbody.integrate(k, q0, p0, [a0, 1.0])

    # shall be within 5% to class solution
    assert_allclose(q[-1] / qt, 1., rtol=0.1)

    # need very high precision for reversibility.
    a, q, p = linearnbody.integrate(k, qt, pt, [1.0, a0], rtol=1e-9)
    a, q, p = linearnbody.integrate(k, q[-1], p[-1], [a0, 1.0], rtol=1e-9)

    assert_allclose(q[-1], qt, rtol=1e-4)
    assert_allclose(p[-1], pt, rtol=1e-4)

