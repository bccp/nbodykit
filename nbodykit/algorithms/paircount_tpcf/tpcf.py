from ..pair_counters import SimulationBoxPairCount, SurveyDataPairCount
from .estimators import LandySzalayEstimator, NaturalEstimator, WedgeBinnedStatistic
from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic

import numpy
import warnings
import logging

class BasePairCount2PCF(object):
    """
    Base class for two-point correlation function algorithms that use
    pair counting. The API largely follows that of
    :class:`~nbodykit.algorithms.SimulationBoxPairCount` and
    :class:`~nbodykit.algorithms.SurveyDataPairCount`.

    Parameters
    ----------
    mode : '1d', '2d', 'projected', 'angular'
        the type of two point correlation function to compute
    data1 : CatalogSource
        the data catalog; must have a 'Position' column
    edges : array_like
        the bin edges along the first binning dimension
    Nmu : int, optional
        when ``mode`` is '2d', the number of mu bins, ranging from 0 to 1
    pimax : float, optional
        when ``mode`` is 'projected', the maximum separation along the line-of-sight
    randoms1 : CatalogSource, optional
        the catalog specifying the un-clustered, random distribution for ``data1``;
        if not provided, analytic randoms will be used
    randoms2 : CatalogSource, optional
        the catalog specifying the un-clustered, random distribution for ``data2``;
        if not provided, analytic randoms will be used
    data2 : CatalogSource, optional
        the second data catalog to cross-correlate; must have a 'Position' column
    R1R2 : SimulationBoxPairCount, SurveyDataPairCount, optional
        if provided, random pairs R1R2 are not recalculated in the Landy-Szalay estimator
    **kws :
        additional keyword arguments passed to the appropriate pair counting class
    """

    def __init__(self, mode, data1, edges,
                    Nmu=None, pimax=None,
                    randoms1=None, randoms2=None, data2=None, R1R2=None, **kws):

        self.comm = data1.comm

        # store the attributes
        self.attrs = {'mode':mode, 'edges':edges, 'Nmu':Nmu, 'pimax':pimax}
        self.attrs.update(kws)

        # store the catalogs
        self.data1 = data1
        self.data2 = data2
        self.randoms1 = randoms1
        self.randoms2 = randoms2
        self.R1R2 = R1R2


    def run(self):
        """
        Run the two-point correlation function algorithm.

        There are two cases here:

        1. If no randoms were provided, and the data is in a simulation box with
           periodic boundary conditions, the natural estimator
           :math:`DD/RR - 1` is used.
        2. If randoms were provided, the Landy-Szalay estimator is used:
           :math:`(D_1 D_2 - D_1 R_2 - D_2 R_1 + R_1 R_2) / R_1 R_2`

        Raises
        ------
        ValueError :
            if periodic boundary conditions were not requested, and ``randoms1``
            is ``None``
        ValueError :
            if periodic boundary conditions were not requested, and
            ``data2`` is not None, but ``randoms2`` is ``None``
        """
        # get the config
        attrs = self.attrs.copy()
        config = attrs.pop('config')
        attrs.update(config)

        # whether we are doing sim volume or mock survey
        if 'periodic' in attrs:
            pair_counter = SimulationBoxPairCount
        else:
            pair_counter = SurveyDataPairCount

        # use analytic randoms for a periodic box
        if 'periodic' in attrs and attrs['periodic'] and self.randoms1 is None:

            if attrs['mode'] == 'angular':
                self.data1 = _restrict_to_spherical_volume(self.data1)
                if self.data2 is not None:
                    self.data2 = _restrict_to_spherical_volume(self.data2)
                if self.comm.rank == 0:
                    msg = "when using analytic randoms for the angular correlation function, "
                    msg += "we restrict the input data to a spherical volume, throwing away objects. "
                    msg += "To use all data, pass in a UniformCatalog as the 'randoms1' keyword"
                    warnings.warn(msg)

            # count the data-data pairs using analytic randoms
            DD = pair_counter(first=self.data1, second=self.data2, **attrs)
            self.R1R2, self.corr = NaturalEstimator(DD)
            self.D1D2 = DD.pairs
            self.D1R2 = self.D2R1 = None

        # need catalog randoms
        else:

            if self.randoms1 is None:
                msg = "a catalog of randoms must be specified as the ``randoms1`` keyword "
                msg += "when the data is not in a simulation box with periodic boundary conditions"
                raise ValueError(msg)

            # use the first randoms for both
            if self.data2 is not None and self.randoms2 is None:
                self.randoms2 = self.randoms1

            # use the Landy-Szalay estimator
            result = LandySzalayEstimator(pair_counter, self.data1, self.data2,
                                            self.randoms1, self.randoms2, R1R2=self.R1R2,
                                            logger=self.logger, **attrs)
            self.D1D2, self.D1R2, self.D2R1, self.R1R2, self.corr = result

    def __getstate__(self):

        # the correlation
        edges = [self.corr.edges[d] for d in self.corr.dims]
        state = {'corr':self.corr.data, 'dims':self.corr.dims, 'edges':edges}

        # the pair counts
        for pc in ['D1D2', 'D1R2', 'D2R1', 'R1R2', 'wp']:
            if getattr(self, pc, None) is not None:
                state[pc] = getattr(self, pc).data
            else:
                state[pc] = None

        state['attrs'] = self.attrs
        return state

    def __setstate__(self, state):

        edges = state.pop('edges')
        dims = state.pop('dims')
        self.__dict__.update(state)

        self.corr = WedgeBinnedStatistic(dims, edges, self.corr)
        if self.wp is not None:
            # NOTE: only edges[0], second dimension was summed over
            self.wp = WedgeBinnedStatistic(dims[:1], edges[:1], self.wp)

        for pc in ['D1D2', 'D1R2', 'D2R1', 'R1R2']:
            val = getattr(self, pc)
            if val is not None:
                setattr(self, pc, WedgeBinnedStatistic(dims, edges, val))

    def save(self, output):
        """
        Save result as a JSON file with name ``output``
        """
        import json
        from nbodykit.utils import JSONEncoder

        # only the master rank writes
        if self.comm.rank == 0:
            self.logger.info('measurement done; saving result to %s' % output)

            with open(output, 'w') as ff:
                json.dump(self.__getstate__(), ff, cls=JSONEncoder)

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, output, comm=None):
        """
        Load a result has been saved to disk with :func:`save`.
        """
        import json
        from nbodykit.utils import JSONDecoder
        if comm.rank == 0:
            with open(output, 'r') as ff:
                state = json.load(ff, cls=JSONDecoder)
        else:
            state = None
        state = comm.bcast(state)
        self = object.__new__(cls)
        self.__setstate__(state)
        self.comm = comm
        return self


class SimulationBox2PCF(BasePairCount2PCF):
    r"""
    Compute the two-point correlation function for data in a simulation box
    as a function of :math:`r`, :math:`(r,\mu)`, :math:`(r_p, \pi)`, or
    :math:`\theta` using pair counting.

    This uses analytic randoms when using periodic conditions, unless
    a randoms catalog is specified. The "natural" estimator (DD/RR-1) is
    used in the former case, and the Landy-Szalay estimator (DD/RR - 2DR/RR + 1)
    in the latter case.

    .. note::
        When using analytic randoms, the expected counts are assumed to
        be unweighted.

    Parameters
    ----------
    mode : '1d', '2d', 'projected', 'angular'
        the type of two-point correlation function to compute; see the Notes below
    data1 : CatalogSource
        the data catalog; must have a 'Position' column
    edges : array_like
        the separation bin edges along the first coordinate dimension;
        depending on ``mode``, the options are :math:`r`, :math:`r_p`, or
        :math:`\theta`. Expected units for distances are :math:`\mathrm{Mpc}/h`
        and degrees for angles. Length of nbins+1
    Nmu : int, optional
        the number of :math:`\mu` bins, ranging from 0 to 1; requred if
        ``mode='2d'``
    pimax : float, optional
        The maximum separation along the line-of-sight when ``mode='projected'``.
        Distances along the :math:`\pi` direction are binned with unit
        depth. For instance, if ``pimax=40``, then 40 bins will be created
        along the :math:`\pi` direction.
    data2 : CatalogSource, optional
        the second data catalog to cross-correlate; must have a 'Position' column
    randoms1 : CatalogSource, optional
        the catalog specifying the un-clustered, random distribution for ``data1``;
        if not provided, analytic randoms will be used
    randoms2 : CatalogSource, optional
        the catalog specifying the un-clustered, random distribution for ``data2``;
        if not provided, analytic randoms will be used
    R1R2 : SimulationBoxPairCount, optional
        if provided, random pairs R1R2 are not recalculated in the Landy-Szalay estimator
    periodic : bool, optional
        whether to use periodic boundary conditions
    BoxSize : float, 3-vector, optional
        the size of the box; if 'BoxSize' is not provided in the source
        'attrs', it must be provided here
    los : 'x', 'y', 'z'; int, optional
        the axis of the simulation box to treat as the line-of-sight direction;
        this can be provided as string identifying one of 'x', 'y', 'z' or
        the equivalent integer number of the axis
    weight : str, optional
        the name of the column in the source specifying the particle weights
    show_progress : bool, optional
        if ``True``, perform the pair counting calculation in 10 iterations,
        logging the progress after each iteration; this is useful for
        understanding the scaling of the code
    **config : key/value pairs
        additional keywords to pass to the :mod:`Corrfunc` function

    Notes
    -----
    This class can compute correlation functions using several different
    coordinate choices, based on the value of the input argument ``mode``.
    The choices are:

    * ``mode='1d'`` : compute pairs as a function of the 3D separation :math:`r`
    * ``mode='2d'`` : compute pairs as a function of the 3D separation :math:`r`
      and the cosine of the angle to the line-of-sight, :math:`\mu`
    * ``mode='projected'`` : compute pairs as a function of distance perpendicular
      and parallel to the line-of-sight, :math:`r_p` and :math:`\pi`
    * ``mode='angular'`` : compute pairs as a function of angle on the sky, :math:`\theta`

    If ``mode='projected'``, the projected correlation function :math:`w_p(r_p)`
    is also computed, using the input :math:`\pi_\mathrm{max}` value.
    """
    logger = logging.getLogger('SimulationBox2PCF')

    def __init__(self, mode, data1, edges, Nmu=None, pimax=None,
                    data2=None, randoms1=None, randoms2=None, R1R2=None,
                    periodic=True, BoxSize=None, los='z',
                    weight='Weight', show_progress=False, **config):

        # format the input arguments
        args = dict(locals())
        args.pop('self')

        # init the base class
        BasePairCount2PCF.__init__(self, **args)

        # and run
        self.run()

    def run(self):
        """
        Run the two-point correlation function algorithm. This attaches
        the following attributes:

        - :attr:`SimulationBox2PCF.D1D2`
        - :attr:`SimulationBox2PCF.D1R2`
        - :attr:`SimulationBox2PCF.D2R1`
        - :attr:`SimulationBox2PCF.R1R2`
        - :attr:`SimulationBox2PCF.corr`
        - :attr:`SimulationBox2PCF.wp` (if ``mode='projected'``)

        Attributes
        ----------
        D1D2 : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the data1 - data2 pair counts
        D1R2 : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the data1 - randoms2 pair counts
        D2R1 : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the data2 - randoms1 pair counts
        R1R2 : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the randoms1 - randoms2 pair counts
        corr : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the correlation function values, stored as the ``corr`` variable,
            computed from the pair counts
        wp : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the projected correlation function, :math:`w_p(r_p)`, computed
            if ``mode='projected'``; correlation is stored as the ``corr`` variable

        Notes
        -----
        The :attr:`D1D2`, :attr:`D1R2`, :attr:`D2R1`, and :attr:`R1R2`
        attributes are identical to the
        :attr:`~nbodykit.algorithms.SimulationBoxPairCount.pairs` attribute
        of :class:`~nbodykit.algorithms.SimulationBoxPairCount`.
        """
        # this does most of the work
        BasePairCount2PCF.run(self)

        # compute wp(rp) if we computed xi(rp, pi)
        if self.attrs['mode'] == 'projected':
            self.wp = _compute_wp(self.corr)


class SurveyData2PCF(BasePairCount2PCF):
    r"""
    Compute the two-point correlation function for observational survey data
    as a function of :math:`r`, :math:`(r,\mu)`, :math:`(r_p, \pi)`, or
    :math:`\theta` using pair counting.

    The Landy-Szalay estimator (DD/RR - 2 DD/RR + 1) is used to transform
    pair counts in to the correlation function.

    Parameters
    ----------
    mode : '1d', '2d', 'projected', 'angular'
        the type of two-point correlation function to compute; see the Notes below
    data1 : CatalogSource
        the data catalog; must have a 'Position' column
    randoms1 : CatalogSource
        the catalog specifying the un-clustered, random distribution for ``data1``
    edges : array_like
        the separation bin edges along the first coordinate dimension;
        depending on ``mode``, the options are :math:`r`, :math:`r_p`, or
        :math:`\theta`. Expected units for distances are :math:`\mathrm{Mpc}/h`
        and degrees for angles. Length of nbins+1
    cosmo : :class:`~nbodykit.cosmology.cosmology.Cosmology`, optional
        the cosmology instance used to convert redshift into comoving distance;
        this is required for all cases except ``mode='angular'``
    Nmu : int, optional
        the number of :math:`\mu` bins, ranging from 0 to 1; requred if
        ``mode='2d'``
    pimax : float, optional
        The maximum separation along the line-of-sight when ``mode='projected'``.
        Distances along the :math:`\pi` direction are binned with unit
        depth. For instance, if ``pimax=40``, then 40 bins will be created
        along the :math:`\pi` direction.
    data2 : CatalogSource, optional
        the second data catalog to cross-correlate; must have a 'Position' column
    randoms2 : CatalogSource, optional
        the catalog specifying the un-clustered, random distribution for ``data2``;
        if not specified and ``data2`` is provied, then ``randoms1`` will be used
        for both.
    R1R2 : SurveyDataPairCount, optional
        if provided, random pairs R1R2 are not recalculated in the Landy-Szalay estimator
    ra : str, optional
        the name of the column in the source specifying the
        right ascension coordinates in units of degrees; default is 'RA'
    dec : str, optional
        the name of the column in the source specifying the declination
        coordinates; default is 'DEC'
    redshift : str, optional
        the name of the column in the source specifying the redshift
        coordinates; default is 'Redshift'
    weight : str, optional
        the name of the column in the source specifying the object weights
    show_progress : bool, optional
        if ``True``, perform the pair counting calculation in 10 iterations,
        logging the progress after each iteration; this is useful for
        understanding the scaling of the code
    **config : key/value pairs
        additional keywords to pass to the :mod:`Corrfunc` function

    Notes
    -----
    This class can compute correlation functions using several different
    coordinate choices, based on the value of the input argument ``mode``.
    The choices are:

    * ``mode='1d'`` : compute pairs as a function of the 3D separation :math:`r`
    * ``mode='2d'`` : compute pairs as a function of the 3D separation :math:`r`
      and the cosine of the angle to the line-of-sight, :math:`\mu`
    * ``mode='projected'`` : compute pairs as a function of distance perpendicular
      and parallel to the line-of-sight, :math:`r_p` and :math:`\pi`
    * ``mode='angular'`` : compute pairs as a function of angle on the sky, :math:`\theta`

    If ``mode='projected'``, the projected correlation function :math:`w_p(r_p)`
    is also computed, using the input :math:`\pi_\mathrm{max}` value.
    """
    logger = logging.getLogger('SurveyData2PCF')

    def __init__(self, mode, data1, randoms1, edges, cosmo=None,
                    Nmu=None, pimax=None, data2=None, randoms2=None, R1R2=None,
                    ra='RA', dec='DEC', redshift='Redshift', weight='Weight',
                    show_progress=False, **config):

        # format the input arguments
        args = dict(locals())
        args.pop('self')

        # init the base class
        BasePairCount2PCF.__init__(self, **args)

        # and run
        self.run()

    def run(self):
        """
        Run the two-point correlation function algorithm. This attaches
        the following attributes:

        - :attr:`SurveyData2PCF.D1D2`
        - :attr:`SurveyData2PCF.D1R2`
        - :attr:`SurveyData2PCF.D2R1`
        - :attr:`SurveyData2PCF.R1R2`
        - :attr:`SurveyData2PCF.corr`
        - :attr:`SurveyData2PCF.wp` (if ``mode='projected'``)

        Attributes
        ----------
        D1D2 : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the data1 - data2 pair counts
        D1R2 : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the data1 - randoms2 pair counts
        D2R1 : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the data2 - randoms1 pair counts
        R1R2 : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the randoms1 - randoms2 pair counts
        corr : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the correlation function values, stored as the ``corr`` variable,
            computed from the pair counts
        wp : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            the projected correlation function, :math:`w_p(r_p)`, computed
            if ``mode='projected'``; correlation is stored as the ``corr`` variable

        Notes
        -----
        The :attr:`D1D2`, :attr:`D1R2`, :attr:`D2R1`, and :attr:`R1R2`
        attributes are identical to the
        :attr:`~nbodykit.algorithms.SurveyDataPairCount.pairs` attribute
        of :class:`~nbodykit.algorithms.SurveyDataPairCount`.
        """
        # this does most of the work
        BasePairCount2PCF.run(self)

        # compute wp(rp) if we computed xi(rp, pi)
        if self.attrs['mode'] == 'projected':
            self.wp = _compute_wp(self.corr)


def _compute_wp(corr):
    r"""
    Compute the projected correlation function :math:`w_p(r_p)` from
    :math:`\xi(r_p, \pi)`.
    """
    # compute wp(rp)
    dpi = numpy.diff(corr.edges['pi'])
    wp = (2*corr['corr']*dpi).sum(axis=-1)

    # return a BinnedStatistic
    toret = corr.copy()
    if len(toret.dims) > 1:
        toret = toret.average(toret.dims[-1])
    toret['corr'] = wp

    return toret

def _restrict_to_spherical_volume(source):
    """
    Slice the input source such that it only includes a spherical volume
    instead the simulation box.

    This returns a CatalogSource only containing objects within a radius
    of 1/2 the minimum box side length of the box center.
    """
    # compute the distance from box center
    origin = 0.5 * source.attrs['BoxSize']
    pos = source['Position'] - origin
    r = (pos**2).sum(axis=-1)**0.5

    # restrict to sphere less than half of box size
    keep = r < 0.5 * source.attrs['BoxSize'].min()
    return source[keep]
