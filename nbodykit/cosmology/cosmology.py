from classylss.binding import ClassEngine, Background, Spectra, Perturbs, Primordial, Thermo
from classylss.astropy_compat import AstropyCompat

import numpy
from six import string_types
import os

class Cosmology(object):
    r"""
    A cosmology calculator based on the CLASS binding in :mod:`classylss`.

    It is a collection of all method provided by the CLASS interfaces.

    The individual interfaces can be accessed too, such that
    `c.Spectra.get_transfer` and `c.get_transfer` are identical.

    .. important::

        A default set of units is assumed. Those units are:

        * temperature: :math:`\mathrm{K}`
        * distance: :math:`h^{-1} \mathrm{Mpc}`
        * wavenumber: :math:`h \mathrm{Mpc}^{-1}`
        * power: :math:`h^{-3} \mathrm{Mpc}^3`
        * density: :math:`10^{10} (M_\odot/h) (Mpc/h)^{-3}`
        * neutrino mass: :math:`\mathrm{eV}`
        * time: :math:`\mathrm{Gyr}`
        * H0: :math:`(h^{-1} \mathrm{Mpc}) / (\mathrm{km/s})`

    Notes
    -----
    * The default configuration assumes a flat cosmology, :math:`\Omega_{0,k}=0`.
      Pass ``Omega_k`` in the ``extra`` keyword dictionary to change this value.
    * By default, a cosmological constant is assumed, with its density value
      inferred by the curvature condition.
    * Non-cosmological constant dark energy can be used by specifying the
      ``w0_fld``, ``wa_fld``, and/or ``Omega_fld`` values.
    * The ``sigma8`` attribute can be set to the desired value, and the internal
      value of ``A_s`` will be automatically adjusted.

    Parameters
    ----------
    h : float
        the dimensionaless Hubble parameter
    T_cmb : float
        the temperature of the CMB in Kelvins
    Omega_b : float
        the current baryon density parameter, :math:`\Omega_{b,0}`
    Omega_cdm : float
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
        should  be passed for no massive neutrinso. The default is a single
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
        extra keyword parameters to pass to CLASS; users should be wary
        of configuration options that may conflict with the base set
        of parameters
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

    def __init__(self,
            h=0.67556,
            T_cmb=2.7255,
            Omega_b=0.022032/0.67556**2,
            Omega_cdm=0.12038/0.67556**2,
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

        # use set state to de-serialize the object.
        self.__setstate__((args,kwargs))

    def __iter__(self):
        """
        Allows dict() to be used on class.
        """
        args = self.args.copy()
        args.update(self.kwargs)
        for k in args:
            yield k, args[k]

    def __dir__(self):
        """ a list of all members from all delegate classes """
        r = []
        # first allow tab completion of delegate names; to help resolve conflicts
        r.extend([n.__name__ for n in self.dro])
        # then allow tab completion of all delegate methods
        for i in reversed(self.dro):
            r.extend(dir(i))
        return sorted(list(set(r)))

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
        return (self.args, self.kwargs)

    @property
    def sigma8(self):
        """
        The amplitude of matter fluctuations at :math:`z=0` in a sphere
        of radius :math:`r = 8 \ h^{-1}\mathrm{Mpc}`.

        This is not an input CLASS parameter, but users can set this parameter
        and the scalar amplitude ``A_s`` will be internally adjusted to
        achieve the desired ``sigma8``.
        """
        return self.Spectra.sigma8

    @sigma8.setter
    def sigma8(self, value):
        if not numpy.isclose(self.sigma8, value):
            set_sigma8(self, value, inplace=True)

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

        print(cls)
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
            extra keyword parameters to pass when initializing

        Returns
        -------
        :class:`Cosmology` :
            the initialized cosmology object
        """
        args = astropy_to_dict(cosmo)
        args.update(kwargs)
        return Cosmology(**args)

    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Initialize a :class:`Cosmology` object from the CLASS parameter file

        Parameters
        ----------
        filename : str
            the name of the parameter file to read
        """
        from classylss import load_ini

        # extract dictionary of parameters from the file
        pars = load_ini(filename)
        pars.update(**kwargs)

        # initialize the engine as the backup delegate.
        toret = object.__new__(cls)
        toret.engine = ClassEngine(pars)
        toret.delegates = {ClassEngine: toret.engine}

        # reconstruct the correct __init__ params
        args, kwargs = sanitize_class_params(toret, pars)

        toret.args = args
        toret.kwargs = kwargs
        return toret

    def __setstate__(self, state):

        # remember for serialization
        self.args, self.kwargs = state

        # verify and set defaults
        pars = verify_parameters(self.args, self.kwargs)

        # initialize the engine as the backup delegate.
        self.engine = ClassEngine(pars)
        self.delegates = {ClassEngine: self.engine}

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
        # initialize a new object (so we have sanitized args/kwargs)
        new = Cosmology(**kwargs)

        # the named keywords
        args = self.args.copy()
        args.update(new.args)

        # the extra keywords
        kwargs = self.kwargs.copy()
        kwargs.update(new.kwargs)
        args.update(kwargs)

        return Cosmology(**args)


def astropy_to_dict(cosmo):
    """
    Convert an astropy cosmology object to a dictionary of parameters
    suitable for initializing a Cosmology object.
    """
    from astropy import cosmology, units

    pars = {}
    pars['h'] = cosmo.h
    pars['T_cmb'] = cosmo.Tcmb0.value
    if cosmo.Ob0 is not None:
        pars['Omega_b'] = cosmo.Ob0
    else:
        raise ValueError("please specify a value 'Ob0' ")
    pars['Omega_cdm'] = cosmo.Om0 - cosmo.Ob0 # should be okay for now

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
        pars['N_ur'] = (cosmo.Neff/3.046) * N_ur[N_massive-1]

        pars['m_ncdm'] = [k.value for k in sorted(m_nu[m_nu > 0.], reverse=True)]
    else:
        pars['m_ncdm'] = []
        pars['N_ur'] = cosmo.Neff

    # specify the curvature
    pars['Omega_k'] = cosmo.Ok0

    # handle dark energy
    if isinstance(cosmo, cosmology.LambdaCDM):
        pass
    elif isinstance(cosmo, cosmology.wCDM):
        pars['w0_fld'] = cosmo.w0
        pars['wa_fld'] = 0.
        pars['Omega_Lambda'] = 0. # use Omega_fld
    elif isinstance(cosmo, cosmology.w0waCDM):
        pars['w0_fld'] = cosmo.w0
        pars['wa_fld'] = cosmo.wa
        pars['Omega_Lambda'] = 0. # use Omega_fld
    else:
        cls = cosmo.__class__.__name__
        valid = ["LambdaCDM", "wCDM", "w0waCDM"]
        msg = "dark energy equation of state not recognized for class '%s'; " %cls
        msg += "valid classes: %s" %str(valid)
        raise ValueError(msg)

    return pars

def verify_parameters(args, extra):
    """
    Verify the input parameters to a :class:`Cosmology` object and
    set various default values.
    """
    # check for conflicts
    for par in CONFLICTS:
        for p in CONFLICTS[par]:
            if p in extra:
                raise ValueError("input parameter conflict; use '%s', not '%s'" %(par, p))

    pars = {}
    pars.update(args)
    pars.update(extra)

    # set some default parameters
    pars.setdefault('output', "vTk dTk mPk")
    pars.setdefault('extra metric transfer functions', 'y')

    # no massive neutrinos
    if pars.get('m_ncdm', None) is None:
        pars['m_ncdm'] = []

    # a single massive neutrino
    if numpy.isscalar(pars['m_ncdm']):
        pars['m_ncdm'] = [pars['m_ncdm']]

    # needs to be a list
    if not isinstance(pars['m_ncdm'], (list,numpy.ndarray)):
        raise TypeError("``m_ncdm`` should be a list of mass values in eV")

    # check gauge
    if pars.get('gauge', 'synchronous') not in ['synchronous', 'newtonian']:
        raise ValueError("'gauge' should be 'synchronous' or 'newtonian'")

    for m in pars['m_ncdm']:
        if m == 0:
            raise ValueError("A zero mass is specified in the non-cold dark matter list. "
                             "This is not needed, as we automatically set N_ur based on "
                             "the number of entries in m_ncdm such that Neff = 3.046.")

    # remove None's -- use CLASS default
    for key in list(pars.keys()):
        if pars[key] is None: pars.pop(key)

    # set cosmological constant to zero if we got fluid w0/wa
    if 'w0_fld' in pars or 'wa_fld' in pars:
        if pars.get('Omega_Lambda', 0) > 0:
            raise ValueError(("non-zero fOmega_Lambda (cosmological constant) specified as "
                             "well as fluid w0/wa; use Omega_fld instead"))
        pars['Omega_Lambda'] = 0.

    # turn on verbosity
    verbose = pars.pop('verbose', False)
    if verbose:
        for par in ['input', 'background', 'thermodynamics', 'perturbations',
                    'transfer', 'primordial', 'spectra', 'nonlinear', 'lensing']:
            name = par + '_verbose'
            if name not in pars: pars[name] = 1

    # maximum k value
    if 'P_k_max_h/Mpc' not in pars:
        pars['P_k_max_h/Mpc'] = pars.pop('P_k_max', 10.)

    # maximum redshift
    if 'z_max_pk' not in pars:
        pars['z_max_pk'] = pars.pop('P_z_max', 100.)

    # nonlinear power?
    if 'non linear' not in pars:
        if pars.pop('nonlinear', False):
            pars['non linear'] = 'halofit'

    # number of massive neutrino species
    pars['N_ncdm'] = len(pars['m_ncdm'])

    # m_ncdm only needed if we have massive neutrinos
    if not pars['N_ncdm']: pars.pop('m_ncdm')

    # from CLASS notes:
    # one more remark: if you have respectively 1,2,3 massive neutrinos,
    # if you stick to the default value pm equal to 0.71611, designed to give m/omega of
    # 93.14 eV, and if you want to use N_ur to get N_eff equal to 3.046 in the early universe,
    # then you should pass here respectively 2.0328,1.0196,0.00641
    N_ur_table = [3.046, 2.0328, 1.0196, 0.00641]
    if 'N_ur' not in pars:
        pars['N_ur'] = N_ur_table[pars['N_ncdm']]

    return pars


def set_sigma8(cosmo, sigma8, inplace=False):
    """
    Return a clone of the input Cosmology object, with the ``sigma8`` value
    set to the specified value.

    Parameters
    ----------
    cosmo : Cosmology
        the input cosmology object
    sigma8 : float
        the desired sigma8 value
    inplace : bool, optional
        if ``True``, update sigma8 of the input ``cosmo`` object, else return
        a new Cosmology object
    """
    # the new scalar amplitude A_s
    A_s = cosmo.A_s * (sigma8/cosmo.sigma8)**2

    # the extra keywords
    kwargs = cosmo.kwargs.copy()

    # set the desired A_s and remove conflicting parameters
    kwargs['A_s'] = A_s
    kwargs.pop('ln10^{10}A_s', None)

    if inplace:
        cosmo.__setstate__((cosmo.args, kwargs))
    else:
        # add the name keywords
        kwargs.update(cosmo.args)

        # new cosmo clone
        cosmo = Cosmology(**kwargs)

    return cosmo

def sanitize_class_params(cosmo, pars):
    """
    Given a dictionary of CLASS parameters, construct the ``args``
    dict and ``kwargs`` dict that can used to initialize a
    Cosmology class, accounting for any possible conflicts.

    The ``args`` dict holds the main (named) __init__ keywords, and the
    ``kwargs`` holds all of the extra keywords.
    """
    args = {}
    kwargs = pars.copy()

    # loop over all parameters
    for name in list(kwargs.keys()):

        # parameter is a main parameter
        if name in CONFLICTS:
            kwargs.pop(name)
            alias = ALIASES.get(name, name) # check for attribute alias
            args[name] = getattr(cosmo, alias)
        else:
            # check if parameter conflicts with main parameter
            for c in CONFLICTS:
                if name in CONFLICTS[c]:
                    kwargs.pop(name)
                    alias = ALIASES.get(c, c)
                    args[c] = getattr(cosmo, alias)

    # set all named keywords that do not have parameter conflicts
    args['gauge'] = cosmo.gauge

    return args, kwargs

# dict mapping input CLASS params to the Cosmology attribute name
ALIASES = {'Omega_b': 'Omega0_b', 'Omega_cdm':'Omega0_cdm', 'T_cmb':'T0_cmb',
           'P_k_max': 'k_max_for_pk'}

# dict that defines input parameters that conflict with each other
CONFLICTS = {'h': ['H0', '100*theta_s'],
             'T_cmb': ['Omega_g', 'omega_g'],
             'Omega_b': ['omega_b'],
             'N_ur': ['Omega_ur', 'omega_ur'],
             'Omega_cdm': ['omega_cdm'],
             'm_ncdm': ['Omega_ncdm', 'omega_ncdm'],
             'P_k_max': ['P_k_max_h/Mpc', 'P_k_max_1/Mpc'],
             'P_z_max': ['z_max_pk'],
             'nonlinear' : ['non linear']
            }
