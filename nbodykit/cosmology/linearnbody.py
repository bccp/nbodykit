import numpy
#from scipy.integrate import odeint
from scipy.integrate import solve_ivp

class LinearNbody:
    """ 
        Perturbations of matter under matter only interaction on an expanding cosmology
        background. Ignoring interaction with radiation.

        This can be considered the linear order perturbative model of hydro nbody-simulations.

        For technical notes, see https://www.overleaf.com/read/pqsgzjssmptb

        Parameters
        ----------
        c_b : float
            baryon velocity in km/s. set to zero to evolve as cdm. Related
            to the thermal history of baryons. 

        background : object
            an object with attributes:
                efunc(z), Omega_b(z), Omega_cdm(z), Omega_ncdm(z) and m_ncdm
            For example, a Cosmology object from nbodykit will work here.

        c_ncdm_1ev_z0 : float
            neutrino velocity in km/s at z=0 for 1ev. The default number 134.423 is widely used.
            set to zero to evolve as cdm.  `c_ncdm = cncdm_1ev_z0 / a / m_ncdm`

    """
    def __init__(self, background, c_b=0, c_ncdm_1ev_z0=134.423):

        self.background = background
        self.c_b = c_b
        self.c_ncdm_1ev_z0 = c_ncdm_1ev_z0

    def efunc(self, z):
        return self.background.efunc(z)

    def Omega_b(self, z):
        return self.background.Omega_b(z)

    def Omega_cdm(self, z):
        return self.background.Omega_cdm(z)

    def Omega_ncdm(self, z):
        return self.background.Omega_ncdm(z)

    @property
    def m_ncdm(self):
        return self.background.m_ncdm

    def integrate(self, k, q0, p0, a, rtol=1e-4):
        """ 
            Solve the 3 fluid model from initial position and momentum q0, and p0, at
            times a.

            This can be slow if neutrino velocity and baryon velocity
            are non-zero. (200,000+ function evaluations with RK45, when rtol=1e-7)

            Parameters
            ----------
            a : array_like
                scale factor requesting the output.

            k : array_like,
                 k values to compute the solution, in h/Mpc. From classylss's transfer function.

            q0 : array_like (Nk, 3)
                initial position, -d, produced by `seed` from CLASSylss transfer function
                three species are cdm, baryon and ncdm.
            p0 : array_like (Nk, 3)
                initial momentum, a * v, from `seed`. 
                three species are cdm, baryon and ncdm.
            rtol : float
                relative accuracy. It appears 1e-4 is good for k ~ 10 with reasonable velocities.

            Returns
            -------
            a : array_like (Na), very close but not in general identical to input a.

            q : array_like (Na, Nk, 3)
                position (-d) at different requested a's
            p : array_like (Na, Nk, 3)
                momentum (a * v) at different requested a's. v is peculiar velocity.
        """
        def marshal(q0, p0):
            vector = numpy.concatenate([q0.ravel(), p0.ravel()], axis=0)
            return vector
        # internally work with loga
        lna = numpy.log(a)

        def func(lna, vector):
            a = numpy.exp(lna)
            q = vector[: len(vector) // 2].reshape(q0.shape)
            p = vector[len(vector) // 2:].reshape(p0.shape)
            z = 1. / a - 1.
            dlna = 1.

            E = (self.efunc(z))
            dt = dlna / E
            dp = - numpy.einsum('kij,kj->ki', self.J(k, a), q) * dt

            dq = p / a ** 2 * dt
            #print(a, dp[0] / q[0], p[0] / q[0])
            return marshal(dq, dp)

        v0 = marshal(q0, p0)
        r = solve_ivp(func, (lna[0], lna[-1]), v0,
                           method='RK45', t_eval=lna,
                           rtol=rtol, atol=0, vectorized=False)
        q = r.y[: len(v0) // 2, ...].reshape(list(q0.shape) + [-1])
        p = r.y[len(v0) // 2:, ...].reshape(list(p0.shape) + [-1])
        #print('nfev=', r.nfev)
        return numpy.exp(r.t), q.transpose((2, 0, 1)), p.transpose((2, 0, 1))

    def J(self, k, a):
        """ The potential term. Use internally in the ODE.
        """
        z = 1./ a - 1.
        E = self.efunc(z)

        Ocdm = self.Omega_cdm(z)
        Ob = self.Omega_b(z)
        Oncdm = self.Omega_ncdm(z)

        if self.c_ncdm_1ev_z0 > 0:
            c_ncdm = (self.c_ncdm_1ev_z0 / self.m_ncdm[0] / a)
        else:
            c_ncdm = 0

        c_b = self.c_b

        j1 = -1.5 * E ** 2 * a ** 2 * numpy.array([
                [Ocdm, Ob, Oncdm,],
                [Ocdm, Ob, Oncdm,],
                [Ocdm, Ob, Oncdm,],
            ])[..., None]

        H0 = 100 # velocity is in km/s, k is in Mpc/h, so H0 is 100.
        j2 = numpy.diag([0, c_b ** 2, c_ncdm **2])[..., None] * k ** 2 / H0 ** 2

        return (j1 + j2).transpose((2, 0, 1))


    @staticmethod
    def seed_from_synchronous(cosmology, a0, Tk=None):
        """
            Convert synchronuous gauge velocity (h_prime) to the momentum in n-body gauge.

            This function repacks the columns to cdm, baryon and ncdm for both q and p, such
            that at a = a0,

            .. code::

                q = - d

                p = a v = - a dd / dt

            Parameters
            ----------
            cosmology : object, Cosmology.
                the cosmology object to obtain hubble (with dimension of time unit)
            a0 : float
                the scaling factor to seed, 0.01 for z=99
            Tk : structured array
                use a precomputed transfer function, must be
                the same format as the output of Cosmology.get_transfer(),
                with 'k' in h/Mpc units, 'd_cdm', 'd_b', 'd_ncdm[0]' and
                'h_prime'.

            Returns
            -------
            k: array of (Nk)
            q0: array of (Nk, 3)
            p0: array of (Nk, 3)

        """
        if Tk is None:
            if cosmology.gauge != 'synchronuous':
                cosmology = cosmology.clone(gauge='synchronous')
            Tk = cosmology.get_transfer(1 / a0 - 1)

        # this requires the cosmology object has the same unit of hubble_function
        # as that of potential

        # use CLASS's units to obtain dd/dt.
        H0 = cosmology.hubble_function(0)

        q0 = -numpy.vstack([Tk['d_cdm'], Tk['d_b'], Tk['d_ncdm[0]']]).T
        p0 = numpy.vstack([  0.5 * Tk['h_prime'] * a0 / H0 ,
                      Tk['t_b'] +  0.5 * Tk['h_prime'] * a0 / H0,
                      Tk['t_ncdm[0]'] + 0.5 * Tk['h_prime'] * a0 / H0,
                  ]).T

        return Tk['k'], q0, p0

