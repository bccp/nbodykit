from nbodykit.cosmology import Cosmology

class HODModel(object):
    """
    A class providing an interface beteween HOD models in
    :mod:`nbodykit` and :mod:`halotools`.

    See the static method :func:`to_halotools` to create a specified
    :mod:`halotools` HOD model given:

    - cosmology
    - redshift
    - mass definition

    .. note::
        Here, mass definition is used to place satellites in halos using
        a NFW profile in order to convert mass to radius.
    """
    @staticmethod
    def to_halotools(cosmo, redshift, mdef, concentration_key=None, **kwargs):
        """
        Return a {func_name} HOD model in the format of :mod:`halotools`.

        See :func:`halotools.empirical_models.{name}_model_dictionary`.

        Parameters
        ----------
        cosmo :
            the nbodykit or astropy Cosmology object to use in the model
        redshift : float
            the desired redshift of the model
        mdef : str, optional
            string specifying mass definition, used for computing default
            halo radii and concentration; should be 'vir' or 'XXXc' or
            'XXXm' where 'XXX' is an int specifying the overdensity
        concentration_key : str
            the name of the column that will specify concentration; if not
            provided, the analytic formula from
            `Dutton and Maccio 2014 <https://arxiv.org/abs/1402.7073>`_
            is used.
        **kwargs :
            additional keywords passed to the model components; see the
            Halotools documentation for further details

        Returns
        -------
        :class:`~halotools.empirical_models.HodModelFactory`
            the halotools object implementing the HOD model
        """
        raise NotImplementedError

# NOTE: we are making zheng 07 separately due to astropy/halotools#827
class Zheng07Model(HODModel):

    @staticmethod
    def to_halotools(cosmo, redshift, mdef, concentration_key=None, **kwargs):
        """
        Return the Zheng 07 HOD model.

        See :func:`halotools.empirical_models.zheng07_model_dictionary`.

        Parameters
        ----------
        cosmo :
            the nbodykit or astropy Cosmology object to use in the model
        redshift : float
            the desired redshift of the model
        mdef : str, optional
            string specifying mass definition, used for computing default
            halo radii and concentration; should be 'vir' or 'XXXc' or
            'XXXm' where 'XXX' is an int specifying the overdensity
        concentration_key : str
            the name of the column that will specify concentration; if not
            provided, the analytic formula from
            `Dutton and Maccio 2014 <https://arxiv.org/abs/1402.7073>`_
            is used.
        **kwargs :
            additional keywords passed to the model components; see the
            Halotools documentation for further details

        Returns
        -------
        :class:`~halotools.empirical_models.HodModelFactory`
            the halotools object implementing the HOD model
        """
        from halotools.empirical_models import Zheng07Sats, Zheng07Cens, NFWPhaseSpace, TrivialPhaseSpace
        from halotools.empirical_models import HodModelFactory

        kwargs.setdefault('modulate_with_cenocc', True)

        # need astropy Cosmology
        if isinstance(cosmo, Cosmology):
            cosmo = cosmo.to_astropy()

        # determine concentration key
        if concentration_key is None:
            conc_mass_model = 'dutton_maccio14'
        else:
            conc_mass_model = 'direct_from_halo_catalog'

        # determine mass column
        mass_key = 'halo_m' + mdef

        # occupation functions
        cenocc = Zheng07Cens(prim_haloprop_key=mass_key, **kwargs)
        satocc = Zheng07Sats(prim_haloprop_key=mass_key, cenocc_model=cenocc, **kwargs)
        satocc._suppress_repeated_param_warning = True

        # profile functions
        kwargs.update({'cosmology':cosmo, 'redshift':redshift, 'mdef':mdef})
        censprof = TrivialPhaseSpace(**kwargs)
        satsprof = NFWPhaseSpace(conc_mass_model=conc_mass_model, **kwargs)

        # make the model
        model = {}
        model['centrals_occupation'] = cenocc
        model['centrals_profile'] = censprof
        model['satellites_occupation'] = satocc
        model['satellites_profile'] = satsprof
        return HodModelFactory(**model)

def HODModelFactory(name, func_name):
    """
    Factory to generate the functions that will return one of the
    pre-built Halotools HOD models.

    Parameters
    ----------
    name : str
        a name of a pre-built HOD model name from Halotools

    Returns
    -------
    callable :
        a function that can compute the specified Halotools model from
        the input cosmology, redshift, mass definition, etc.
    """
    import textwrap

    def to_halotools(cosmo, redshift, mdef, concentration_key=None, **kwargs):
        """
        Parameters
        ----------
        cosmo :
            the nbodykit or astropy Cosmology object to use in the model
        redshift : float
            the desired redshift of the model
        mdef : str, optional
            string specifying mass definition, used for computing default
            halo radii and concentration; should be 'vir' or 'XXXc' or
            'XXXm' where 'XXX' is an int specifying the overdensity
        concentration_key : str
            the name of the column that will specify concentration; if not
            provided, the analytic formula from
            `Dutton and Maccio 2014 <https://arxiv.org/abs/1402.7073>`_
            is used.
        **kwargs :
            additional keywords passed to the model components; see the
            Halotools documentation for further details

        Returns
        -------
        :class:`~halotools.empirical_models.HodModelFactory`
            the halotools object implementing the HOD model
        """
        from halotools.empirical_models import PrebuiltHodModelFactory

        # modulate sats with cen occ model by default
        kwargs.setdefault('modulate_with_cenocc', True)

        # need astropy Cosmology
        if isinstance(cosmo, Cosmology):
            cosmo = cosmo.to_astropy()

        # determine concentration key
        if concentration_key is None:
            conc_mass_model = 'dutton_maccio14'
        else:
            conc_mass_model = 'direct_from_halo_catalog'

        # determine mass column
        mass_key = 'halo_m' + mdef

        # the configuration
        kwargs.update({'cosmology':cosmo, 'redshift':redshift, 'mdef':mdef, 'prim_haloprop_key':mass_key})
        return PrebuiltHodModelFactory(name, **kwargs)

    # make the doc string
    hdr = "Return the {func_name} HOD model from Halotools.\n\n".format(func_name=func_name)
    hdr += "See :func:`halotools.empirical_models.{func_name}_model_dictionary` for further details.\n"
    to_halotools.__doc__ = hdr + textwrap.dedent(to_halotools.__doc__)

    # make the new class object and return it
    newclass = type(name, (HODModel,),{"to_halotools": staticmethod(to_halotools), "__doc__":__doc__})
    return newclass

# generate the specialized models
Leauthaud11Model = HODModelFactory('leauthaud11', 'Leauthaud11Model')
Hearin15Model    = HODModelFactory('hearin15', 'Hearin15Model')

__all__ = ['Zheng07Model', 'Leauthaud11Model', 'Hearin15Model']
