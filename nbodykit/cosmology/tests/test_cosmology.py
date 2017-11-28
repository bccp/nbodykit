from nbodykit.cosmology import Cosmology, Planck15, WMAP9
from numpy.testing import assert_allclose, assert_array_equal
import numpy
import pytest

def test_old_Omega_syntax():

    c1 = Cosmology(Omega_b=0.04)
    c2 = Cosmology(Omega0_b=0.04)
    assert c1.Omega0_b == c2.Omega0_b

    c1 = Cosmology(T_cmb=2.7)
    c2 = Cosmology(T0_cmb=2.7)
    assert c1.T0_cmb == c2.T0_cmb

    c1 = Cosmology(Omega0_k=0.05)
    c2 = Cosmology(Omega_k=0.05)
    assert c1.Omega0_k == c2.Omega0_k

    c1 = Cosmology(Omega0_lambda=0.7)
    c2 = Cosmology(Omega_lambda=0.7)
    c3 = Cosmology(Omega0_Lambda=0.7)
    assert c1.Omega0_lambda == c2.Omega0_lambda
    assert c1.Omega0_lambda == c3.Omega0_lambda

def test_deprecated_init():

    # all valid deprecated kwargs
    with pytest.warns(FutureWarning):
        c1 = Cosmology(H0=67.6, Om0=0.31, flat=True)
        c2 = Cosmology(H0=67.6, Om0=0.31, Ode0=0.7, flat=False, w0=-0.9)

    # missing valid and deprecated
    with pytest.raises(Exception):
        c = Cosmology(h=0.7, flat=True)

    # passing arguments and mixing
    with pytest.raises(Exception):
        c = Cosmology(0.7, flat=True)

    # parameter conflict
    with pytest.raises(Exception):
        c3 = Cosmology(H0=70., flat=True, h=0.7)

    assert_allclose(c1.h, 0.676)
    assert_allclose(c2.h, 0.676)
    assert_allclose(c1.Om0, 0.31)
    assert_allclose(c2.Om0, 0.31)
    assert_allclose(c1.Ok0, 0.)
    assert_allclose(c2.Ode0, 0.7)
    assert_allclose(c2.w0_fld, -0.9)

def test_efunc_prime():
    epsilon = 1e-4
    z = numpy.linspace(0, 3, 1000) + epsilon

    # cosmology with no massive neutrinos
    cosmo = WMAP9
    d1 = cosmo.efunc_prime(z)
    d2 = (cosmo.efunc(z+epsilon) - cosmo.efunc(z-epsilon))/(2*epsilon) * -(1+z)**2
    assert_allclose(d1, d2, err_msg="efunc_prime error for cosmo with no massive neutrinos", rtol=1e-3)

    # cosmology with massive neutrinos
    cosmo = Planck15
    d1 = cosmo.efunc_prime(z)
    d2 = (cosmo.efunc(z+epsilon) - cosmo.efunc(z-epsilon))/(2*epsilon) * -(1+z)**2
    assert_allclose(d1, d2, err_msg="efunc_prime error for cosmo with massive neutrinos", rtol=1e-3)

def test_load_precision():
    from classylss import load_precision

    p = load_precision('pk_ref.pre')
    c = Cosmology(gauge='synchronous', tol_background_integration=1e-5, **p)
    assert_allclose(c.Omega_cdm(0), c.Omega0_cdm)

def test_clone():
    c = Cosmology(gauge='synchronous', tol_background_integration=1e-5)
    c2 = c.clone(Omega0_b=0.04)
    assert_allclose(c2.Omega0_b, 0.04)
    c2 = c2.clone()
    assert_allclose(c2.Omega0_b, 0.04)

def test_from_file():

    import tempfile, pickle
    with tempfile.NamedTemporaryFile(mode='w') as ff:
        ff.write("H0=70\nomega_b = 0.0266691\nomega_cdm = 0.110616\nT_cmb=2.7255\n")
        ff.seek(0)

        # load from a file and check values
        c = Cosmology.from_file(ff.name)
        assert_allclose(c.Omega0_b*c.h**2, 0.0266691)
        assert_allclose(c.Omega0_cdm*c.h**2, 0.110616)

    # clone
    c2 = c.clone(Omega0_b=0.04)
    assert_allclose(c2.Omega0_b, 0.04)

    # serialize and make sure we get the same
    s = pickle.dumps(c)
    c1 = pickle.loads(s)

    assert_allclose(c.Omega0_cdm, c1.Omega0_cdm)
    assert_allclose(c.Omega0_b, c1.Omega0_b)

def test_conflicts():

    # h is the correct param
    with pytest.raises(Exception):
        c = Cosmology(h=0.7, H0=70)

    with pytest.raises(Exception):
        c = Cosmology(Omega0_b=0.04, Omega_b=0.04)

    # Omega_b is the correct param
    with pytest.raises(Exception):
        c = Cosmology(Omega0_b=0.04, omega_b=0.02)

def test_unknown_params():

    # warn about unknown parameters
    with pytest.warns(UserWarning):
        c = Cosmology(unknown_paramter=100.)

def test_set_sigma8():

    # set sigma8 by adjusting A_s internally
    c = Cosmology().match(sigma8=0.80)

    # run CLASS and compute sigma8
    assert_allclose(c.sigma8, 0.80)

def test_set_Omega0_cb():

    c = Cosmology().match(Omega0_cb=0.4)

    assert_allclose(c.Omega0_cb, 0.4)

    c = Cosmology().match(Omega0_m=0.4)
    assert_allclose(c.Omega0_m, 0.4)


def test_sigma8_z():

    z = numpy.linspace(0, 1, 100)
    c = Cosmology()

    s8_z = c.sigma8_z(z)
    D_z = c.scale_independent_growth_factor(z)
    assert_allclose(s8_z, D_z*c.sigma8, rtol=1e-3)


def test_cosmology_sane():
    c = Cosmology(gauge='synchronous', verbose=True)

    assert_allclose(c.Omega_cdm(0), c.Omega0_cdm)
    assert_allclose(c.Omega_g(0), c.Omega0_g)
    assert_allclose(c.Omega_b(0), c.Omega0_b)
    assert_allclose(c.Omega_ncdm(0), c.Omega0_ncdm)
    assert_allclose(c.Omega_ur(0), c.Omega0_ur)
    assert_allclose(c.Omega_ncdm(0), c.Omega0_ncdm_tot)

    assert_allclose(c.Omega_pncdm(0), c.Omega0_pncdm)
    assert_allclose(c.Omega_m(0), c.Omega0_m)
    assert_allclose(c.Omega_r(0), c.Omega0_r)

    # total density in 10e10 Msun/h unit
    assert_allclose(c.rho_tot(0), 27.754999)

    # comoving distance to z=1.0 in Mpc/h unit.
    assert_allclose(c.comoving_distance(1.0), 3396.157391 * c.h)

    # conformal time in Mpc unit.
    assert_allclose(c.tau(1.0), 3396.157391)

    assert_allclose(c.efunc(0), 1.) # hubble in Mpc/h km/s unit
    assert_allclose(c.efunc(0) - c.efunc(1 / 0.9999 - 1),
                    0.0001 * c.efunc_prime(0), rtol=1e-3)

def test_cosmology_density():
    c = Cosmology(gauge='synchronous')
    z = [0, 1, 2, 5, 9, 99]
    assert_allclose(c.rho_cdm(z), c.Omega_cdm(z) * c.rho_tot(z))
    assert_allclose(c.rho_g(z), c.Omega_g(z) * c.rho_tot(z))
    assert_allclose(c.rho_ncdm(z), c.Omega_ncdm(z) * c.rho_tot(z))
    assert_allclose(c.rho_b(z), c.Omega_b(z) * c.rho_tot(z))
    assert_allclose(c.rho_m(z), c.Omega_m(z) * c.rho_tot(z))
    assert_allclose(c.rho_r(z), c.Omega_r(z) * c.rho_tot(z))
    assert_allclose(c.rho_ur(z), c.Omega_ur(z) * c.rho_tot(z))

def test_cosmology_vect():
    c = Cosmology(gauge='synchronous')

    assert_allclose(c.Omega_cdm([0]), c.Omega0_cdm)

    assert_array_equal(c.Omega_cdm([]).shape, [0])
    assert_array_equal(c.Omega_cdm([0]).shape, [1])
    assert_array_equal(c.Omega_cdm([[0]]).shape, [1, 1])

    assert_array_equal(c.rho_k([[0]]).shape, [1, 1])

    k, z = numpy.meshgrid([0, 1], [0.01, 0.05, 0.1, 0.5], sparse=True, indexing='ij')

    pk = c.get_pk(z=z, k=k)
    assert_array_equal(pk.shape, [2, 4])

def test_cosmology_a_max():
    c = Cosmology(gauge='synchronous', a_max=2.0)
    print(c.parameter_file)
    assert c.a_max == 2.0
    t = c.Omega_m(-0.1)
    t = c.efunc(-0.1)
    t = c.scale_independent_growth_factor(-0.1)

    #t = c.get_transfer(z=-0.1)

def test_cosmology_transfer():
    c = Cosmology()
    t = c.get_transfer(z=0)
    assert 'h_prime' in t.dtype.names
    assert 'k' in t.dtype.names
    assert 'd_cdm' in t.dtype.names

def test_cosmology_get_pk():
    c = Cosmology()
    p = c.get_pk(z=0, k=0.1)
    p1 = c.Spectra.get_pk(z=0, k=0.1)

    # ensure the dro did use get_pk of Spectra rather than that from Primordial
    assert_allclose(p, p1)

def test_to_astropy():

    from astropy.cosmology import FlatLambdaCDM, LambdaCDM
    from astropy.cosmology import FlatwCDM, wCDM
    from astropy.cosmology import Flatw0waCDM, w0waCDM

    # lambda CDM
    for cls in [FlatLambdaCDM, LambdaCDM]:
        if "Flat" in cls.__name__:
            c1 = Cosmology(Omega_k=0.)
        else:
            c1 = Cosmology(Omega_k=0.05)
        c2 = c1.to_astropy()
        assert isinstance(c2, cls)
        assert_allclose(c2.Ok0, c1.Omega0_k, rtol=1e-3)

    # w0 CDM
    for cls in [FlatwCDM, wCDM]:
        if "Flat" in cls.__name__:
            c1 = Cosmology(w0_fld=-0.9, Omega_k=0.)
        else:
            c1 = Cosmology(w0_fld=-0.9, Omega_k=0.05)
        c2 = c1.to_astropy()
        assert isinstance(c2, cls)
        assert_allclose(c2.Ok0, c1.Omega0_k, rtol=1e-3)
        assert_allclose(c2.w0, c1.w0_fld)

    # wa w0 CDM
    for cls in [Flatw0waCDM, w0waCDM]:
        if "Flat" in cls.__name__:
            c1 = Cosmology(wa_fld=0.05, w0_fld=-0.9, Omega_k=0.)
        else:
            c1 = Cosmology(wa_fld=0.05, w0_fld=-0.9, Omega_k=0.05)
        c2 = c1.to_astropy()
        assert isinstance(c2, cls)
        assert_allclose(c2.Ok0, c1.Omega0_k, rtol=1e-3)
        assert_allclose(c2.w0, c1.w0_fld)
        assert_allclose(c2.wa, c1.wa_fld)

def test_from_astropy():

    from astropy.cosmology import FlatLambdaCDM, LambdaCDM
    from astropy.cosmology import FlatwCDM, wCDM
    from astropy.cosmology import Flatw0waCDM, w0waCDM

    # LambdaCDM
    flat = {'H0':70, 'Om0':0.3, 'Ob0':0.04, 'Tcmb0':2.7}
    for cls in [FlatLambdaCDM, LambdaCDM]:
        if "Flat" in cls.__name__:
            x = cls(**flat)
        else:
            x = cls(Ode0=0.75, **flat)
        c = Cosmology.from_astropy(x)
        assert_allclose(c.Ok0, x.Ok0)
        assert_allclose(c.Omega0_fld, 0.) # Omega0_lambda is nonzero
        assert_allclose(c.Odm0, x.Odm0)

    # w0 CDM
    for cls in [FlatwCDM, wCDM]:
        if "Flat" in cls.__name__:
            x = cls(w0=-0.9, **flat)
        else:
            x = cls(w0=-0.9, Ode0=0.75, **flat)
        c = Cosmology.from_astropy(x)
        assert_allclose(c.Ok0, x.Ok0)
        assert_allclose(c.Odm0, x.Odm0)
        assert_allclose(c.w0, x.w0)
        assert_allclose(c.Omega0_lambda, 0.) # Omega_fld is nonzero

    # w0,wa CDM
    for cls in [Flatw0waCDM, w0waCDM]:
        if "Flat" in cls.__name__:
            x = cls(w0=-0.9, wa=0.01, **flat)
        else:
            x = cls(w0=-0.9, wa=0.01, Ode0=0.75, **flat)
        c = Cosmology.from_astropy(x)
        assert_allclose(c.Ok0, x.Ok0)
        assert_allclose(c.Odm0, x.Odm0)
        assert_allclose(c.w0, x.w0)
        assert_allclose(c.wa, x.wa)
        assert_allclose(c.Omega0_lambda, 0.) # Omega_fld is nonzero

def test_immutable():

    c = Cosmology()
    with pytest.raises(ValueError):
        c.A_s = 2e-9 # immutable

    # can add non-CLASS attributes still
    c.test = 'TEST'

def test_bad_no_Ob0():

    from astropy.cosmology import FlatLambdaCDM
    c = FlatLambdaCDM(Om0=0.3, H0=70) # no Ob0

    with pytest.raises(ValueError):
        c = Cosmology.from_astropy(c)

def test_bad_astropy_class():

    from astropy.cosmology import w0wzCDM
    c = w0wzCDM(Om0=0.3, H0=70, Ode0=0.7) # no Ob0

    with pytest.raises(ValueError):
        c = Cosmology.from_astropy(c)

def test_massive_neutrinos():

    # single massive neutrino
    c = Cosmology(m_ncdm=0.6)
    assert c.N_ncdm == 1

    # do not need 0 values
    with pytest.raises(ValueError):
        c = Cosmology(m_ncdm=[0.6, 0.])

def test_no_massive_neutrinos():

    c = Cosmology(m_ncdm=None)
    assert c.has_massive_nu == False

def test_bad_input():

    with pytest.raises(ValueError):
        c = Cosmology(gauge='BAD')

    # specifying w0_fld + Omega_Lambda is inconsistent
    with pytest.raises(ValueError):
        c = Cosmology(Omega_Lambda=0.7, w0_fld=-0.9)


def test_cosmology_dir():
    c = Cosmology()
    d = dir(c)
    assert "Background" in d
    assert "Spectra" in d
    assert "Omega0_m" in d

def test_cosmology_pickle():
    import pickle
    c = Cosmology()
    s = pickle.dumps(c)
    c1 = pickle.loads(s)
    assert c1.parameter_file == c.parameter_file

def test_cosmology_clone():
    c = Cosmology(gauge='synchronous')

    c1 = Cosmology(gauge='newtonian')
    assert 'newtonian' in c1.parameter_file

    c2 = Cosmology(P_k_max=1.01234567)
    assert '1.01234567' in c2.parameter_file

def test_astropy_compat():
    c = Cosmology(gauge='synchronous', m_ncdm=[0.06])

    assert_allclose(c.Odm(0), c.Odm0)
    assert_allclose(c.Ogamma(0), c.Ogamma0)
    assert_allclose(c.Ob(0), c.Ob0)
    assert_allclose(c.Onu(0), c.Onu0)
    assert_allclose(c.Ok(0), c.Ok0)
    assert_allclose(c.Ode(0), c.Ode0)
    assert_array_equal(c.has_massive_nu, True)
