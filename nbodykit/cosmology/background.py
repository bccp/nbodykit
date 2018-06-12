import numpy as np
from scipy.integrate import odeint

class Perturbation:
    """
    Perturbation Growth coefficients at several orders.

    2-LPT is implemented. This implements the single fluid
    model of Boltamann equations. Therefore it is accurate
    only in a matter dominated universe.

    All derivatives are against ``lna``.

    .. note::
        Formulas are derived from Yin Li's notes on 2LPT.

    """

    def __init__(self, a, a_normalize=1.0):

        if a is None:
            lna = np.log(np.logspace(-7, 0, 1024*10, endpoint=True))
        else:
            a = np.array(a, copy=True).ravel() # ensure this is 1-d
            if a_normalize not in a: # ensure redshift 0 is in the list, for normalization
                a = np.concatenate([[a_normalize], a])
            a.sort()
            if a[0] > 1e-7: # add a high redshift starting point.
                a = np.concatenate([[1e-7], a])
            lna = np.log(a)

        self.lna = lna

        y0 = self.get_initial_condition()

        self._D1, self._D2 = self._solve(y0, a_normalize)


    def D1(self, a, order=0):
        """ Linear order growth function.

            Parameters
            ----------
            a : float, array_like
                scaling factor
            order : int
                order of differentation; 1 for first derivative against log a.

            Returns
            -------
            array_like : linear order growth function.
        """
        lna = np.log(a)
        return np.interp(lna, self.lna, self._D1[:, order])

    def D2(self, a, order=0):
        """ Second order growth function.

            Parameters
            ----------
            a : float, array_like
                scaling factor
            order : int
                order of differentation; 1 for first derivative against log a.

            Returns
            -------
            array_like : second order growth function.
        """
        lna = np.log(a)
        return np.interp(lna, self.lna, self._D2[:, order])

    def f1(self, a):
        """ Linear order growth rate

            Parameters
            ----------
            a : float, array_like
                scaling factor
            order : int
                order of differentation; 1 for first derivative against log a.

            Returns
            -------
            array_like : linear order growth rate.
        """
        return self.D1(a, order=1) / self.D1(a, order=0)

    def f2(self, a):
        """ Second order growth rate.

            Parameters
            ----------
            a : float, array_like
                scaling factor
            order : int
                order of differentation; 1 for first derivative against log a.

            Returns
            -------
            array_like : second order growth rate.
        """
        return self.D2(a, order=1) / self.D2(a, order=0)

    def Gp(self, a):
        """
        FastPM growth factor function, eq, 19
        """
        return self.D1(a)

    def Gp2(self, a):
        """ Gp for second order LPT
        FastPM growth factor function, eq, 19
         """
        return self.D2(a)

    def gp(self, a):
        """
        Notice the derivative of D1 is against ln a but gp is d D1 / da, so
        gp = D1(a, order=1) / a
        """
        return self.D1(a, order=1) / a


    def gp2(self, a):
        """ gp for second order LPT
        """
        return self.D2(a, order=1) / a

    def Gf(self, a):
        """
        FastPM growth factor function, eq, 20
        """
        return self.D1(a, 1) * a ** 2 * self.E(a)

    def Gf2(self, a):
        """ Gf but for second order LPT
        """
        return self.D2(a, 1) * a ** 2 * self.E(a)

    def gf(self, a):
        """
        Similarly, the derivative is against ln a, so
        gf = Gf(a, order=1) / a
        """
        return 1 / a * (
            self.D1(a, 2) * a ** 2 * self.E(a) \
            +  self.D1(a, 1) * (
                    a ** 2 * self.E(a, order=1)
                +   2 * a ** 2 * self.E(a))
            )
    def gf2(self, a):
        """
            gf but for second order LPT
        """
        return 1 / a * (
            self.D2(a, 2) * a ** 2 * self.E(a) \
            +  self.D2(a, 1) * (
                    a ** 2 * self.E(a, order=1)
                +   2 * a ** 2 * self.E(a))
            )

    def E(self, a, order=0):
        """ Hubble function and derivatives against log a.

        """
        if order == 0:
            return self.efunc(a)
        else:
            return self.efunc_prime(a) * a

    def Hfac(self, a):
        return -2. - self.E(a, order=1) / self.E(a)

    def ode(self, y, lna):
        D1, F1, D2, F2 = y
        a = np.exp(lna)
        hfac = self.Hfac(a)
        omega = self.Om(a)
        F1p = hfac * F1 + 1.5 * omega * D1
        D1p = F1
        F2p = hfac * F2 + 1.5 * omega * D2 - 1.5 * omega * D1 ** 2
        D2p = F2
        return D1p, F1p, D2p, F2p

    def _solve(self, y0, a_normalize):
        # solve with critical point at a=1.0, lna=0.
        y = odeint(self.ode, y0, self.lna, tcrit=[0.], atol=0)

        v1 = []
        v2 = []
        for yi, lnai in zip(y, self.lna):
            D1, F1, D2, F2 = yi
            D1p, F1p, D2p, F2p = self.ode(yi, lnai)
            v1.append((D1, F1, F1p))
            v2.append((D2, F2, F2p))

        v1 = np.array(v1)
        v2 = np.array(v2)

        ind = abs(self.lna - np.log(a_normalize)).argmin()
        # normalization to 1 at a=a_normalize
        v1 /= v1[ind][0]
        v2 /= v2[ind][0]
        return v1, v2

class MatterDominated(Perturbation):
    """

    Perturbation with matter dominated initial condition.

    This is usually referred to the single fluid approximation as well.

    The result here is accurate upto numerical precision. If Omega0_m 

    Parameters
    ----------
    Omega0_m :
        matter density at redshift 0

    Omega0_lambda :
        Lambda density at redshift 0, default None; set to ensure flat universe.

    Omega0_k:
        Curvature density at redshift 0, default : 0

    a : array_like
        a list of time steps where the factors are exact.
        other a values are interpolated.
    """
    def __init__(self, Omega0_m, Omega0_lambda=None, Omega0_k=0, a=None, a_normalize=1.0):
        if Omega0_lambda is None:
            Omega0_lambda = 1 - Omega0_k - Omega0_m

        self.Omega0_lambda = Omega0_lambda
        self.Omega0_m = Omega0_m
        self.Omega0_k = Omega0_k
        # Om0 is added for backward compatiblity
        self.Om0 = Omega0_m
        Perturbation.__init__(self, a, a_normalize)

    def get_initial_condition(self):
        a0 = np.exp(self.lna[0])

        # matter dominated initial conditions
        y0 = [a0, a0, -3./7 * a0**2, -6. / 7 *a0**2]
        return y0

    def efunc(self, a):
        return (self.Omega0_m / a**3+ self.Omega0_k / a**2 + self.Omega0_lambda) ** 0.5

    def efunc_prime(self, a):
        return 0.5 / self.efunc(a) * (-3 * self.Omega0_m / (a**4) + -2 * self.Omega0_k / (a**3))

    def Om(self, a):
        return (self.Omega0_m / a ** 3) / self.efunc(a) ** 2

class RadiationDominated(Perturbation):
    """

    Perturbation with Radiation dominated initial condition

    This is approximated because the single fluid scale independent
    solution will need an initial condition that comes from a
    true Boltzmann code.

    Here, the first order result is tuned to agree
    at sub-percent level comparing to a true multi-fluid
    boltzmann code under Planck15 cosmology.

    Parameters
    ----------
    cosmo: :class:`~nbodykit.cosmology.core.Cosmology`
        a astropy Cosmology like object.

    a : array_like
        a list of time steps where the factors are exact.
        other a values are interpolated.
    """
    def __init__(self, cosmo, a=None, a_normalize=1.0):
#        assert cosmo.Ogamma0 == 0
#        assert cosmo.Onu0 == 0

        self._cosmo = cosmo
        self.Omega0_m = cosmo.Om0

        # Om0 is added for backward compatiblity
        self.Om0 = self.Omega0_m
        self.Omega0_gamma = cosmo.Ogamma0

        Perturbation.__init__(self, a, a_normalize)

    def get_initial_condition(self):
        # suggested by Yin Li, not so sensitive to a_H
        # as long as it is early enough.
        # I used this number to ensure best agreement with CLASS
        # for Planck15. (1e-4 rtol)
        a_H = 5.22281250e-05

        a0 = np.exp(self.lna[0])
        Om = self.Om(a0)

        D1i = np.log(a0 / a_H)
        # radiation dominated initial conditions
        y0 = [
            D1i,
            1.0,
             -1.5 * Om * (D1i ** 2 - 4 * D1i + 6),
             -1.5 * Om * (D1i ** 2 - 2 * D1i + 2)]
        return y0

    def efunc(self, a):
        z = 1. / a - 1.0
        return self._cosmo.efunc(z)

    def efunc_prime(self, a):
        z = 1. / a - 1.0
        return self._cosmo.efunc_prime(z)

    def Om(self, a):
        z = 1./a-1
        return self._cosmo.Omega_b(z) + self._cosmo.Omega_cdm(z) # non-relativistic 


from nbodykit.utils import deprecate

PerturbationGrowth = deprecate("PerturbationGrowth", RadiationDominated)

