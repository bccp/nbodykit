from astropy import cosmology, units
from scipy.integrate import quad
import numpy as np
import functools

def removeunits(f):
    """
    Decorator to remove units from :class:`astropy.units.Quantity`
    instances
    """
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        ans = f(*args, **kwargs)
        if isinstance(ans, units.Quantity):
            ans = ans.value
        return ans
    return wrapped

class fittable(object):
    """
    A "fittable" function

    There exists a `.fit()` method of the original function
    which returns a spline-interpolated version of the function
    for a specified variable
    """
    def __init__(self, func, instance=None):

        # update the docstring, etc from the original func
        functools.update_wrapper(self, func)
        self.func = func
        self.instance = instance

    def __get__(self, instance, owner):
        # descriptor that binds the method to an instance
        return fittable(self.func, instance=instance)

    def __call__(self, *args, **kwargs):
        return self.func(self.instance, *args, **kwargs)

    def fit(self, argname, kwargs={}, bins=1024, range=None):
        """
        Interpolate the function for the given argument (`argname`)
        with a :class:`~scipy.interpolate.InterpolatedUnivariateSpline`

        `range` and `bins` behave like :func:`numpy.histogram`

        Parameters
        ----------
        argname : str
            the name of the variable to interpolate
        kwargs : dict; optional
            dict of keywords to pass to the original function
        bins : int, iterable; optional
            either an iterable specifying the bin edges, or an
            integer specifying the number of linearly-spaced bins
        range : tuple; optional
            the range to fit over if `bins` specifies an integer

        Returns
        -------
        spl : callable
            the callable spline function
        """
        from scipy import interpolate
        from astropy.units import Quantity

        if isiterable(bins):
            bin_edges = np.asarray(bins)
        else:
            assert len(range) == 2
            bin_edges = np.linspace(range[0], range[1], bins + 1, endpoint=True)

        # evaluate at binned points
        d = {}
        d.update(kwargs)
        d[argname] = bin_edges
        y = self.__call__(**d)

        # preserve the return value of astropy functions by attaching
        # the right units to the splined result
        spl = interpolate.InterpolatedUnivariateSpline(bin_edges, y)
        if isinstance(y, Quantity):
            return lambda x: Quantity(spl(x), y.unit)
        else:
            return spl

def vectorize_if_needed(func, *x):
    """
    Helper function to vectorize functions on array inputs;
    borrowed from :mod:`astropy.cosmology.core`
    """
    if any(map(isiterable, x)):
        return np.vectorize(func)(*x)
    else:
        return func(*x)

def isiterable(obj):
    """
    Returns `True` if the given object is iterable;
    borrowed from :mod:`astropy.cosmology.core`
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False

class Cosmology(dict):
    """
    An extension of the :mod:`astropy.cosmology` framework that can
    store additional, orthogonal parameters and behaves like a read-only
    dictionary

    The class relies on :mod:`astropy.cosmology` as the underlying
    "engine" for calculation of cosmological quantities. This "engine"
    is stored as :attr:`engine` and supports :class:`~astropy.cosmology.LambdaCDM`
    and :class:`~astropy.cosmology.wCDM`, and their flat equivalents

    Any attributes or functions of the underlying astropy engine
    can be directly accessed as attributes or keys of this class

    ..note::

    A default set of units is assumed, so attributes stored internally
    as :class:`astropy.units.Quantity` instances will be returned
    here as numpy arrays. Those units are:

        - temperature: ``K``
        - distance: ``Mpc``
        - density: ``g/cm^3``
        - neutrino mass: ``eV``
        - time: ``Gyr``
        - H0: ``Mpc/km/s``

    .. warning::

        This class does not currently support a non-constant dark energy
        equation of state
    """
    def __init__(self, H0=67.6, Om0=0.31, Ob0=0.0486, Ode0=0.69, w0=-1., Tcmb0=2.7255,
                    Neff=3.04, m_nu=0., flat=False, name=None, **kwargs):

        """
        Parameters
        ----------
        H0 : float
            the Hubble constant at z=0, in km/s/Mpc
        Om0 : float
            matter density/critical density at z=0
        Ob0 : float
            baryon density/critical density at z=0
        Ode0 : float
            dark energy density/critical density at z=0
        w0 : float
            dark energy equation of state
        Tcmb0 : float
            temperature of the CMB in K at z=0
        Neff : float
            the effective number of neutrino species
        m_nu : float, array_like
            mass of neutrino species in eV
        flat : bool
            if `True`, automatically set `Ode0` such that `Ok0` is zero
        name : str
            a name for the cosmology
        kwargs :
            additional key/value pairs to store in the dictionary
        """
        # convert neutrino mass to a astropy `Quantity`
        if m_nu is not None:
            m_nu = units.Quantity(m_nu, 'eV')

        # the astropy keywords
        kws = {'name':name, 'Ob0':Ob0, 'w0':w0, 'Tcmb0':Tcmb0, 'Neff':Neff, 'm_nu':m_nu, 'Ode0':Ode0}

        # determine the astropy class
        if w0 == -1.0: # cosmological constant
            cls = 'LambdaCDM'
            kws.pop('w0')
        else:
            cls = 'wCDM'

        # use special flat case if Ok0 = 0
        if flat:
            cls = 'Flat' + cls
            kws.pop('Ode0')

        # initialize the astropy engine
        self.engine = getattr(cosmology, cls)(H0=H0, Om0=Om0, **kws)

        # add valid params to the underlying dict
        for k in kws:
            if hasattr(self.engine, k):
                kwargs[k] = getattr(self.engine, k)
        dict.__init__(self, H0=H0, Om0=Om0, **kwargs)

        # store D_z normalization
        integrand = lambda a: a ** (-3) * self.engine.inv_efunc(1/a-1.) ** 3
        self._Dz_norm = 1. / quad(integrand, 0., 1. )[0]

    def __dir__(self):
        """
        Explicitly the underlying astropy engine's attributes as
        part of the attributes of this class
        """
        this_attrs = set(dict.__dir__(self)) | set(self.keys())
        engine_attrs = set(self.engine.__dir__())
        return list(this_attrs|engine_attrs)

    @classmethod
    def from_astropy(self, cosmo, **kwargs):
        """
        Return a :class:`Cosmology` instance from an astropy cosmology

        Parameters
        ----------
        cosmo : subclass of :class:`astropy.cosmology.FLRW`
            the astropy cosmology instance
        ** kwargs :
            extra key/value parameters to store in the dictionary
        """
        valid = ['H0', 'Om0', 'Ob0', 'Ode0', 'w0', 'Tcmb0', 'Neff', 'm_nu']
        for name in valid:
            if hasattr(cosmo, name):
                kwargs[name] = getattr(cosmo, name)
        kwargs['flat'] = cosmo.Ok0 == 0.
        kwargs.setdefault('name', getattr(cosmo, 'name', None))

        toret = Cosmology(**kwargs)
        toret.__doc__ += "\n" + cosmo.__doc__
        return toret

    def __setitem__(self, key, value):
        """
        No setting --> read-only
        """
        raise ValueError("Cosmology is a read-only dictionary; see clone() to create a copy with changes")

    @removeunits
    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    @removeunits
    def __missing__(self, key):
        """
        Missing dict keys returned only if they are attributes of the
        underlying astropy engine and are not callable functions
        """
        # try to return the parameter from the engine
        if hasattr(self.engine, key):
            toret = getattr(self.engine, key)
            if not callable(toret):
                return toret

        # otherwise fail
        raise KeyError("no such parameter '%s' in Cosmology" %key)

    @removeunits
    def __getattr__(self, key):
        """
        Try to return attributes from the underlying astropy engine and
        then from the dictionary

        Notes
        -----
        For callable attributes part of the astropy engine's public API (i.e.,
        functions that do not begin with a '_'), the function will be decorated
        with the :class:`fittable` class
        """
        try:
            toret = getattr(self.engine, key)
            # if a callable function part of public API of the "engine", make it fittable
            if callable(toret) and not key.startswith('_'):
                toret = fittable(toret.__func__, instance=self)

            return toret
        except:
            if key in self:
                return self[key]
            raise AttributeError("no such attribute '%s'" %key)

    def clone(self, **kwargs):
        """
        Returns a copy of this object, potentially with some changes.

        Returns
        -------
        newcos : Subclass of FLRW
        A new instance of this class with the specified changes.

        Notes
        -----
        This assumes that the values of all constructor arguments
        are available as properties, which is true of all the provided
        subclasses but may not be true of user-provided ones.  You can't
        change the type of class, so this can't be used to change between
        flat and non-flat.  If no modifications are requested, then
        a reference to this object is returned.

        Examples
        --------
        To make a copy of the Planck15 cosmology with a different Omega_m
        and a new name:

        >>> from astropy.cosmology import Planck15
        >>> cosmo = Cosmology.from_astropy(Planck15)
        >>> newcos = cosmo.clone(name="Modified Planck 2013", Om0=0.35)
        """
        # filter out astropy-defined parameters and extras
        extras = {k:self[k] for k in self if not hasattr(self.engine, k)}
        extras.update({k:kwargs.pop(k) for k in list(kwargs) if not hasattr(self.engine, k)})

        # make the new astropy instance
        new_engine = self.engine.clone(**kwargs)

        # return a new Cosmology instance
        return self.from_astropy(new_engine, **extras)

    def efunc_prime(self, z):
        """
        Function giving the derivative of :func:`efunc with respect
        to the scale factor ``a``

        Parameters
        ----------
        z : array-like
            Input redshifts.

        Returns
        -------
        eprime : ndarray, or float if input scalar
            The derivative of the hubble factor redshift-scaling with respect
            to scale factor
        """
        if not np.all(self.de_density_scale(z)==1.0):
            raise NotImplementedError("non-constant dark energy redshift dependence is not supported")

        if isiterable(z):
            z = np.asarray(z)
        zp1 = 1.0 + z

        # compute derivative of Or term wrt to scale factor
        if self.has_massive_nu:
            Or = self.Ogamma0 * (1 + self.nu_relative_density(z))

            # compute derivative of nu_relative_density() function with
            # uses fitting formula from Komatsu et al 2011
            p = 1.83
            invp = 0.54644808743  # 1.0 / p
            k = 0.3173
            curr_nu_y = self._nu_y / (1. + np.expand_dims(z, axis=-1))
            x = (k * curr_nu_y) ** p
            drel_mass_per =  x / curr_nu_y * (1.0 + x) ** (invp-1) * self._nu_y
            drel_mass = drel_mass_per.sum(-1) + self._nmasslessnu
            nu_relative_density_deriv = 0.22710731766 * self._neff_per_nu * drel_mass

            rad_deriv = -4*Or*zp1**5 + zp1**4*self.Ogamma0*nu_relative_density_deriv
        else:
            Or = self.Ogamma0 + self.Onu0
            rad_deriv = -4*Or*zp1**5

        # dE^2 / da (assumes Ode0 independent of a)
        esq_prime = rad_deriv - 3*self.Om0*zp1**4 - 2*self.Ok0*zp1**3

        # dE/dA
        eprime = esq_prime / (2*self.efunc(z))
        return eprime

    @fittable
    def growth_rate(self, z):
        """
        Linear growth rate :math:`f(z) = dlnD / dlna`, where ``a`` is the
        scale factor and ``D`` is the growth function, given by :func:`D_z`

        Parameters
        ----------
        z : array-like
            Input redshifts.

        Returns
        -------
        fz : ndarray, or float if input scalar
            The linear growth rate evaluated at the input redshifts
        """
        from .background import PerturbationGrowth

        z = np.asarray(z)
        a = 1./(1+z)

        pt = PerturbationGrowth(self, a=a)
        return pt.f1(a)

    @fittable
    def growth_function(self, z):
        """
        Linear growth function :math:`D(z)` at redshift ``z``

        .. math::

          D(a) \propto H(a) \int_0^a \frac{da'}{a'^3 H(a')^3}

        The normalization is such that :math:`D_1(a=1) = D_1(z=0) = 1`

        Parameters
        ----------
        z : array-like
            Input redshifts.

        Returns
        -------
        Dz : ndarray, or float if input scalar
            The linear growth function evaluated at the input redshifts
        """
        from .background import PerturbationGrowth

        z = np.asarray(z)
        a = 1./(1+z)

        pt = PerturbationGrowth(self, a=a)
        return pt.D1(a)
