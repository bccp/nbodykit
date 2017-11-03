from classylss.binding import ClassEngine, Background, Spectra, Perturbs, Primordial, Thermo
from classylss.astropy_compat import AstropyCompat

import numpy
from six import string_types
import os
import functools

def store_user_kwargs():
    """
    Decorator that adds the ``_user_kwargs`` attribute to the class to track
    which arguments the user actually supplied.
    """
    def decorator(function):
        @functools.wraps(function)
        def inner(self, *args, **kwargs):
            self._user_args = args
            self._user_kwargs = kwargs
            return function(self, *args, **kwargs)
        return inner
    return decorator


class Cosmology(object):
    r"""
    A cosmology calculator based on the CLASS binding in :mod:`classylss`.

    It is a collection of all method provided by the CLASS interfaces.
    The object is immutable. To obtain an instance with a new set of parameters
    use :func:`clone` or :func:`match`.

    The individual interfaces can be accessed too, such that
    `c.Spectra.get_transfer` and `c.get_transfer` are identical.

    .. important::

        A default set of units is assumed. Those units are:

        * temperature: :math:`\mathrm{K}`
        * distance: :math:`h^{-1} \mathrm{Mpc}`
        * wavenumber: :math:`h \mathrm{Mpc}^{-1}`
        * power: :math:`h^{-3} \mathrm{Mpc}^3`
        * density: :math:`10^{10} (M_\odot/h) (\mathrm{Mpc}/h)^{-3}`
        * neutrino mass: :math:`\mathrm{eV}`
        * time: :math:`\mathrm{Gyr}`
        * :math:`H_0`: :math:`(\mathrm{km} \ \mathrm{s^{-1}}) / (h^{-1} \ \mathrm{Mpc})`

    Notes
    -----
    * The default configuration assumes a flat cosmology, :math:`\Omega_{0,k}=0`.
      Pass ``Omega0_k`` as a keyword to specify the desired non-flat curvature.
    * For consistency of variable names, the present day values can be passed
      with or without '0' postfix, e.g., ``Omega0_cdm`` is translated to
      ``Omega_cdm`` as CLASS always uses the names without `0` as input
      parameters.
    * By default, a cosmological constant (``Omega0_lambda``) is assumed, with
      its density value inferred by the curvature condition.
    * Non-cosmological constant dark energy can be used by specifying the
      ``w0_fld``, ``wa_fld``, and/or ``Omega_fld`` values.
    * To pass in CLASS parameters that are not valid Python argument names, use
      the dictionary/keyword arguments trick, e.g.
      ``Cosmology(..., **{'temperature contributions': 'y'})``
    * ``Cosmology(**dict(c))`` is not supposed to work; use ``Cosmology.from_dict(dict(c))``.

    Parameters
    ----------
    h : float
        the dimensionless Hubble parameter
    T0_cmb : float
        the temperature of the CMB in Kelvins
    Omega0_b : float
        the current baryon density parameter, :math:`\Omega_{b,0}`. Currently
        unrealistic cosmology where Omega_b == 0 is not supported.
    Omega0_cdm : float
        the current cold dark matter density parameter, :math:`\Omega_{cdm,0}`
    N_ur : float
        the number of ultra-relativistic (massless neutrino) species; the
        default number is inferred based on the number of massive neutrinos
        via the following logic: if you have respectively 1,2,3 massive
        neutrinos and use the default ``T_ncdm`` value (0.71611 K), designed
        to give m/omega of 93.14 eV, and you wish to have ``N_eff=3.046`` in
        the early universe, then ``N_ur`` is set to 2.0328, 1.0196, 0.00641,
        respectively.
    m_ncdm : list, None
        the masses (in eV) for all massive neutrino species; an empty list
        should  be passed for no massive neutrinos. The default is a single
        massive neutrino with mass of 0.06 eV
    P_k_max : float
        the maximum ``k`` value to compute power spectrum results to, in units
        of :math:`h/Mpc`
    P_z_max : float
        the maximum redshift to compute power spectrum results to
    gauge : str,
        either synchronous or newtonian
    n_s : float
        the tilt of the primordial power spectrum
    nonlinear : bool
        whether to compute nonlinear power spectrum results via HaloFit
    verbose : bool
        whether to turn on the default CLASS logging for all submodules
    **kwargs :
        extra keyword parameters to pass to CLASS. Mainly used to pass-in
        parameter names that are not valid Python function argument names,
        e.g. ``temperature contributions``, or ``number count contributions``.
        Users should be wary of configuration options that may conflict
        with the base set of parameters. To override parameters, chain the
        result with :func:`clone`.
    """
    # delegate resolve order -- a pun at mro; which in
    # this case introduces the meta class bloat and doesn't solve
    # the issue. We want delayed initialization of interfaces
    # or so-called 'mixins'.
    # easier to just use delegates with a customized getattr.
    # this doesn't work well with automated documentation tools though,
    # unfortunately.

    dro = [AstropyCompat, Thermo, Spectra, Perturbs, Primordial, Background, ClassEngine]
    dro_dict = dict([(n.__name__, n) for n in dro])

    @store_user_kwargs()
    def __init__(self,
            h=0.67556,
            T0_cmb=2.7255,
            Omega0_b=0.022032/0.67556**2,
            Omega0_cdm=0.12038/0.67556**2,
            N_ur=None,
            m_ncdm=[0.06],
            P_k_max=10.,
            P_z_max=100.,
            gauge='synchronous',
            n_s=0.9667,
            nonlinear=False,
            verbose=False,
            **kwargs # additional arguments to pass to CLASS
        ):

        # quickly copy over all arguments --
        # at this point locals only contains the arguments.
        args = dict(locals())

        # store the extra CLASS params
        kwargs = args.pop('kwargs')

        # remove some non-CLASS variables
        args.pop('self')

        # check for deprecated init signature
        deprecated_args = check_deprecated_init(self._user_args, self._user_kwargs)
        if deprecated_args is not None:

            # check for conflicts between named args user passed and
            # deprecated args passed via **kwargs
            for a in deprecated_args:
                if a in self._user_kwargs:
                    raise ValueError("Parameter conflicts; use '%s' parameter only" %a)

            # if we make it here, it is a valid deprecated syntax
            import warnings
            warnings.warn(("This init signature is deprecated; see the Cosmology "
                           "docstring for new signature"), FutureWarning)
            args = deprecated_args
        else:
            # merge the kwargs; without resolving conflicts.
            args.update(kwargs)

        # check for input conflicts (using kwargs user actually input)
        check_args(self._user_kwargs)

        # verify and set defaults
        pars = compile_args(args)

        # use set state to de-serialize the object.
        self.__setstate__(pars)

    def __str__(self):
        """
        Return a dict string when printed
        """
        return dict(self).__str__()
    
    def __iter__(self):
        """
        Allows dict() to be used on class.
        Use :func:`from_dict` to reconstruct an instance.
        """
        pars = self.pars.copy()
        for k in pars:
            yield k, pars[k]

    def __dir__(self):
        """ a list of all members from all delegate classes """
        r = []
        # first allow tab completion of delegate names; to help resolve conflicts
        r.extend([n.__name__ for n in self.dro])
        # then allow tab completion of all delegate methods
        for i in reversed(self.dro):
            r.extend(dir(i))
        return sorted(list(set(r)))

    def __setattr__(self, key, value):

        # do not allow setting of properties of the delegate classes
        if any(hasattr(n, key) for n in self.dro):
            raise ValueError(("the Cosmology object is immutable; use clone() or "
                              "match() to update parameters"))

        return object.__setattr__(self, key, value)

    def __getattr__(self, name):
        """
        Find the proper delegate, initialize it, and run the method
        """
        # getting a delegate explicitly, e.g. c.Background
        if name in self.dro_dict:
            iface = self.dro_dict[name]
            if iface not in self.delegates:
                self.delegates[iface] = iface(self.engine)
            return self.delegates[iface]

        # resolving a name from the delegates : c.Om0 => c.Background.Om0
        for iface in self.dro:
            if hasattr(iface, name):
                if iface not in self.delegates:
                    self.delegates[iface] = iface(self.engine)
                d = self.delegates[iface]
                return getattr(d, name)
        else:
            raise AttributeError("Attribute `%s` not found in any of the delegate objects" % name)

    def __getstate__(self):
        return (self.pars)

    @property
    def sigma8(self):
        """
        The amplitude of matter fluctuations at :math:`z=0` in a sphere
        of radius :math:`r = 8 \ h^{-1}\mathrm{Mpc}`.

        This is not an input CLASS parameter. To scale ``sigma8``, use
        :func:`match`, which adjusts scalar amplitude ``A_s`` to
        achieve the desired ``sigma8``.
        """
        return self.Spectra.sigma8

    @property
    def Omega0_cb(self):
        """
        The total density of CDM and Baryon.

        This is not an input CLASS parameter. To scale ``Omega0_cb``, use
        :func:`match`.
        """
        return self.Background.Omega0_cdm + self.Background.Omega0_b

    def match(self, sigma8=None, Omega0_cb=None, Omega0_m=None):
        """
        Creates a new cosmology that matches a derived parameter. This is different
        from clone, where CLASS parameters are used.

        Note that we only supoort matching one derived parameter at a time,
        because the matching is in general non-commutable.

        Parameters
        ----------
        sigma8 : float or None
            We scale the scalar amplitude ``A_s`` to achieve the desired ``sigma8``.
        Omega0_cb: float or None
            Desired total energy density of CDM and baryon.
        Omega0_m: float or None
            Desired total energy density of matter-like components (included ncdm)

        Returns
        -------
        A new cosmology parameter where the derived parameter matches the given constrain.

        """

        if sum(0 if i is None else 1 for i in [sigma8, Omega0_cb, Omega0_m]) != 1:
            raise ValueError("Only match one derived parameter at one time; but multiple is given.")

        if sigma8 is not None:
            return self.clone(A_s=self.A_s * (sigma8/self.sigma8)**2)

        if Omega0_cb is not None:
            rat = Omega0_cb / self.Omega0_cb
            return self.clone(Omega_b=rat * self.Omega0_b, Omega_cdm=rat * self.Omega0_cdm)

        if Omega0_m is not None:
            Omega0_cb = Omega0_m - (self.Omega0_ncdm_tot - self.Omega0_pncdm_tot) - self.Omega0_dcdm
            return self.match(Omega0_cb=Omega0_cb)

        return self

    def to_astropy(self):
        """
        Initialize and return a subclass of :class:`astropy.cosmology.FLRW`
        from the :class:`Cosmology` class.

        Returns
        -------
        subclass of :class:`astropy.cosmology.FLRW` :
            the astropy class holding the cosmology values
        """
        import astropy.cosmology as ac
        import astropy.units as au

        is_flat = True
        needs_w0 = False
        needs_wa = False

        pars = {}
        pars['H0'] = 100*self.h
        pars['Om0'] = self.Omega0_b + self.Omega0_cdm # exclude massive neutrinos to better match astropy
        pars['Tcmb0'] = self.Tcmb0
        pars['Neff'] = self.Neff
        pars['Ob0'] = self.Ob0

        if self.has_massive_nu:

            # all massless by default
            m_nu = numpy.zeros(int(numpy.floor(self.Neff)))

            # then add massive species
            m_nu[:len(self.m_ncdm)] = self.m_ncdm[:]
            pars['m_nu'] = au.Quantity(m_nu, au.eV)

        if self.Ok0 != 0.:
            pars['Ode0'] = self.Ode0
            is_flat = False

        if self.wa_fld != 0:
            pars['wa'] = self.wa_fld
            pars['w0'] = self.wa_fld
            needs_wa = True

        if self.w0_fld != -1:
            pars['w0'] = self.w0_fld
            needs_w0 = True

        # determine class to return
        prefix = "" if not is_flat else "Flat"
        if needs_wa:
            cls = prefix + "w0waCDM"
        elif needs_w0:
            cls = prefix + "wCDM"
        else:
            cls = prefix + "LambdaCDM"
        cls = getattr(ac, cls)

        return cls(**pars)

    @classmethod
    def from_astropy(kls, cosmo, **kwargs):
        """
        Initialize and return a :class:`Cosmology` object from a subclass of
        :class:`astropy.cosmology.FLRW`.

        Parameters
        ----------
        cosmo : subclass of :class:`astropy.cosmology.FLRW`.
            the astropy cosmology instance
        **kwargs :
            extra keyword parameters to pass when initializing;
            they shall not be in conflict with the parameters
            inferred from cosmo. To override parameters,
            chain the result with :func:`clone`.

        Returns
        -------
        :class:`Cosmology` :
            the initialized cosmology object
        """
        args = astropy_to_dict(cosmo)
        # merge in additional arguments -- this will die if
        # there are conflicts.
        args.update(kwargs)

        # astropy_to_dict creates args, so we can use the 'user-friendly'
        # constructor.
        return Cosmology(**args)

    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Initialize a :class:`Cosmology` object from the CLASS parameter file

        Parameters
        ----------
        filename : str
            the name of the parameter file to read
        **kwargs :
            extra keyword parameters to pass when initializing;
            they shall not be in conflict with the parameters
            inferred from cosmo. To override parameters,
            chain the result with :func:`clone`.
        """
        from classylss import load_ini

        # extract dictionary of parameters from the file
        pars = load_ini(filename)

        # intentionally not using merge; use clone if
        # parameters are to modified.
        pars.update(kwargs)

        return cls.from_dict(pars)

    @classmethod
    def from_dict(kls, pars):
        """
        Creates a Cosmology from a pars dictionary.

        This is a rather 'raw' API.
        The dictionary must be readable by ClassEngine.
        Unlike ``Cosmology(**args)``, ``pars`` must
        not contain any convenient names defined here.
        """
        self = object.__new__(Cosmology)
        self.__setstate__(pars)
        return self

    def __setstate__(self, state):

        pars = state

        # initialize the engine as the backup delegate.
        self.engine = ClassEngine(pars)
        self.delegates = {ClassEngine: self.engine}
        self.pars = pars

    def clone(self, **kwargs):
        """
        Create a new cosmology based on modification of self, with the
        input keyword parameters changed.

        Parameters
        ----------
        **kwargs :
            keyword parameters to adjust

        Returns
        -------
        :class:`Cosmology`
            a copy of self, with the input ``kwargs`` adjusted
        """
        # this call to merge_args is OK because self.pars is
        # a valid set of args
        args = merge_args(self.pars, kwargs)
        check_args(args)
        pars = compile_args(args)

        return type(self).from_dict(pars)

def astropy_to_dict(cosmo):
    """
    Convert an astropy cosmology object to a dictionary of parameters
    suitable for initializing a Cosmology object.
    """
    from astropy import cosmology, units

    args = {}
    args['h'] = cosmo.h
    args['T0_cmb'] = cosmo.Tcmb0.value
    if cosmo.Ob0 is not None:
        args['Omega0_b'] = cosmo.Ob0
    else:
        raise ValueError("please specify a value 'Ob0' ")
    args['Omega0_cdm'] = cosmo.Om0 - cosmo.Ob0 # should be okay for now

    # handle massive neutrinos
    if cosmo.has_massive_nu:

        # convert to eV
        m_nu = cosmo.m_nu
        if hasattr(m_nu, 'unit') and m_nu.unit != units.eV:
            m_nu = m_nu.to(units.eV)
        else:
            m_nu = units.eV * m_nu
        # from CLASS notes:
        # one more remark: if you have respectively 1,2,3 massive neutrinos,
        # if you stick to the default value pm equal to 0.71611, designed to give m/omega of
        # 93.14 eV, and if you want to use N_ur to get N_eff equal to 3.046 in the early universe,
        # then you should pass here respectively 2.0328,1.0196,0.00641
        N_ur = [2.0328, 1.0196, 0.00641]
        N_massive = (m_nu > 0.).sum()
        args['N_ur'] = (cosmo.Neff/3.046) * N_ur[N_massive-1]

        args['m_ncdm'] = [k.value for k in sorted(m_nu[m_nu > 0.], reverse=True)]
    else:
        args['m_ncdm'] = []
        args['N_ur'] = cosmo.Neff

    # specify the curvature
    args['Omega0_k'] = cosmo.Ok0

    # handle dark energy
    if isinstance(cosmo, cosmology.LambdaCDM):
        pass
    elif isinstance(cosmo, cosmology.wCDM):
        args['w0_fld'] = cosmo.w0
        args['wa_fld'] = 0.
        args['Omega0_Lambda'] = 0. # use Omega_fld
    elif isinstance(cosmo, cosmology.w0waCDM):
        args['w0_fld'] = cosmo.w0
        args['wa_fld'] = cosmo.wa
        args['Omega0_Lambda'] = 0. # use Omega_fld
    else:
        cls = cosmo.__class__.__name__
        valid = ["LambdaCDM", "wCDM", "w0waCDM"]
        msg = "dark energy equation of state not recognized for class '%s'; " %cls
        msg += "valid classes: %s" %str(valid)
        raise ValueError(msg)

    return args

def compile_args(args):
    """
    Compile the input args of Cosmology object to the input parameters (pars) to
    a :class:`Cosmology` object.

    A variety of defaults are set to tune CLASS for quantities used in
    large scale structures.

    Difference between pars and args:
     - anything that is valid pars is also valid args.
     - after replacing our customizations in args, we get pars.

    Note that CLASS will check for additional conflicts.

    see :func:`merge_args`
    """
    pars = {} # we try to make pars write only.

    # set some default parameters
    pars.setdefault('output', "vTk dTk mPk")
    pars.setdefault('extra metric transfer functions', 'y')

    # args and pars are pretty much compatible;
    pars.update(args)

    def set_alias(pars_name, args_name):
        if args_name not in args: return
        v = args[args_name]
        pars.pop(args_name) # pop because we copied everything.
        if pars_name in args:
            v = args[pars_name]
        pars[pars_name] = v

    set_alias('T_cmb', 'T0_cmb')
    set_alias('Omega_cdm', 'Omega0_cdm')
    set_alias('Omega_b', 'Omega0_b')
    set_alias('Omega_k', 'Omega0_k')
    set_alias('Omega_ur', 'Omega0_ur')
    set_alias('Omega_Lambda', 'Omega_lambda') # classylss variable has lowercase l
    set_alias('Omega_Lambda', 'Omega0_lambda') # classylss variable has lowercase l
    set_alias('Omega_Lambda', 'Omega0_Lambda')
    set_alias('Omega_fld', 'Omega0_fld')
    set_alias('Omega_ncdm', 'Omega0_ncdm')
    set_alias('Omega_g', 'Omega0_g')

    # turn on verbosity
    if 'verbose' in args:
        pars.pop('verbose')
        verbose = args['verbose']
        if verbose:
            for par in ['input', 'background', 'thermodynamics', 'perturbations',
                        'transfer', 'primordial', 'spectra', 'nonlinear', 'lensing']:
                name = par + '_verbose'
                if name not in pars: pars[name] = 1

    # no massive neutrinos
    if 'm_ncdm' in args:
        pars.pop('m_ncdm')
        m_ncdm = args['m_ncdm']
        if m_ncdm is None:
            m_ncdm = []

        if numpy.isscalar(m_ncdm):
            # a single massive neutrino
            m_ncdm = [m_ncdm]

        if isinstance(m_ncdm, (list, numpy.ndarray)):
            m_ncdm = list(m_ncdm)
        else:
            raise TypeError("``m_ncdm`` should be a list of mass values in eV")

        for m in m_ncdm:
            if m == 0:
                raise ValueError("A zero mass is specified in the non-cold dark matter list. "
                                 "This is not needed, as we automatically set N_ur based on "
                                 "the number of entries in m_ncdm such that Neff = 3.046.")

        # number of massive neutrino species
        pars['N_ncdm'] = len(m_ncdm)

        # m_ncdm only needed if we have massive neutrinos
        if len(m_ncdm) > 0:
            pars['m_ncdm'] = m_ncdm

        # from CLASS notes:
        # one more remark: if you have respectively 1,2,3 massive neutrinos,
        # if you stick to the default value pm equal to 0.71611, designed to give m/omega of
        # 93.14 eV, and if you want to use N_ur to get N_eff equal to 3.046 in the early universe,
        # then you should pass here respectively 2.0328,1.0196,0.00641
        N_ur_table = [3.046, 2.0328, 1.0196, 0.00641]
        if args['N_ur'] is None:
            pars['N_ur'] = N_ur_table[len(m_ncdm)]

    if 'N_ur' in args:
        if args['N_ur'] is not None:
            pars['N_ur'] = args['N_ur']

    # check gauge
    if 'gauge' in args:
        if args['gauge'] not in ['synchronous', 'newtonian']:
            raise ValueError("'gauge' should be 'synchronous' or 'newtonian'")

    # set cosmological constant to zero if we got fluid w0/wa
    if 'w0_fld' in args or 'wa_fld' in args:
        if pars.get('Omega_Lambda', 0) > 0:
            raise ValueError(("non-zero Omega_Lambda (cosmological constant) specified as "
                             "well as fluid w0/wa; use Omega_fld instead"))
        pars['Omega_Lambda'] = 0.


    # maximum k value
    set_alias('P_k_max_h/Mpc', 'P_k_max')

    # maximum redshift
    set_alias('z_max_pk', 'P_z_max')

    # nonlinear
    set_alias('non linear', 'nonlinear')
    # sorry we use a boolean but
    # class uses existence of string.
    if pars.pop('non linear', False):
        pars['non linear'] = 'halofit'

    # remove None's for remaining parameters -- None means using a default from CLASS
    # NOTE: do this last since m_ncdm=None means no massive_neutrinos
    for key in list(pars.keys()):
        if pars[key] is None: pars.pop(key)

    return pars


def merge_args(args, moreargs):
    """
    merge moreargs into args.

    Those defined in moreargs takes priority than those
    defined in args.

    see :func:`compile_args`
    """
    args = args.copy()

    for name in moreargs.keys():
        # pop those conflicting with me from the old pars
        for eq in find_eqcls(name):
            if eq in args: args.pop(eq)

    args.update(moreargs)
    return args

def check_deprecated_init(args, kwargs):
    """
    Check if ``kwargs`` uses the (now deprecated) signature of ``Cosmology``
    prior to version 0.2.6.

    If using the deprecated syntax, this returns the necessary arguments for
    the new signature, and ``None`` otherwise.
    """
    from astropy import cosmology, units
    defaults = {'H0':67.6, 'Om0':0.31, 'Ob0':0.0486, 'Ode0':0.69, 'w0':-1.,
                'Tcmb0':2.7255, 'Neff':3.04, 'm_nu':0., 'flat':False}

    # the deprecated kwargs
    deprecated_args = [k for k in kwargs if k in defaults]

    # all clear; nothing to do
    if not len(deprecated_args):
        return

    # if we got deprecated kwargs, make sure we didn't get any valid kwargs!!
    if not all(a in defaults for a in kwargs) or len(kwargs) and len(args):
        msg = "mixing deprecated and valid arguments for the Cosmology class; "
        msg += 'the following args are deprecated: %s' % str(deprecated_args)
        raise ValueError(msg)

    # update old defaults with input params
    defaults.update(kwargs)

    if defaults['m_nu'] is not None:
        defaults['m_nu'] = units.Quantity(defaults['m_nu'], 'eV')

    # determine the astropy class
    if defaults['w0'] == -1.0: # cosmological constant
        cls = 'LambdaCDM'
        defaults.pop('w0')
    else:
        cls = 'wCDM'

    # use special flat case if Ok0 = 0
    if defaults.pop('flat'):
        cls = 'Flat' + cls
        defaults.pop('Ode0')

    # initialize the astropy engine and convert to dict for Cosmology()
    astropy_cosmo = getattr(cosmology, cls)(**defaults)
    return astropy_to_dict(astropy_cosmo)


def check_args(args):
    cf = {}
    for name in args.keys():
        cf[name] = []
        for eq in find_eqcls(name):
            if eq == name: continue
            if eq in args: cf[name].append(eq)

    for name in cf.keys():
        if len(cf[name]) > 0:
            raise ValueError("Conflicted parameters are given: %s" % str(cf))

# dict that defines input parameters that conflict with each other
CONFLICTS = [('h', 'H0', '100*theta_s'),
             ('T_cmb', 'Omega_g', 'omega_g', 'Omega0_g'),
             ('Omega_b', 'omega_b', 'Omega0_b'),
             ('Omega_fld', 'Omega0_fld'),
             ('Omega_Lambda', 'Omega0_Lambda'),
             ('N_ur', 'Omega_ur', 'omega_ur', 'Omega0_ur'),
             ('Omega_cdm', 'omega_cdm', 'Omega0_cdm'),
             ('m_ncdm', 'Omega_ncdm', 'omega_ncdm', 'Omega0_ncdm'),
             ('P_k_max', 'P_k_max_h/Mpc', 'P_k_max_1/Mpc'),
             ('P_z_max', 'z_max_pk'),
             ('nonlinear', 'non linear'),
             ('A_s', 'ln10^{10}A_s'),
            ]

def find_eqcls(key):
    for cls in CONFLICTS:
        if key in cls:
            return cls
    else:
        return ()
