from .base import PairCountBase, verify_input_sources
import numpy
import logging


class SurveyDataPairCount(PairCountBase):
    r"""
    Count (weighted) pairs of objects from a survey data catalog
    as a function of :math:`r`, :math:`(r,\mu)`, :math:`(r_p, \pi)`, or
    :math:`\theta` using the :mod:`Corrfunc` package.

    See the Notes below for the allowed coordinate dimensions.

    The default weighting scheme uses the product of the weights for each
    object in a pair.

    Results are computed when the class is inititalized. See the documenation
    of :func:`~SurveyDataPairCount.run` for the attributes storing the
    results.

    .. note::

        The algorithm expects the positions of particles from a survey catalog
        be the sky coordinates, right ascension and declination, and redshift.
        To compute pair counts in a simulation box using Cartesian
        coordinates, see :class:`~nbodykit.algorithms.SimulationBoxPairCount`.

    .. warning::
        The right ascension and declination columns should be specified
        in degrees.

    Parameters
    ----------
    mode : '1d', '2d', 'projected', 'angular'
        compute pair counts as a function of the specified coordinate basis;
        see the Notes section below for specifics
    first : CatalogSource
        the first source of particles, providing the 'Position' column
    edges : array_like
        the separation bin edges along the first coordinate dimension;
        depending on ``mode``, the options are :math:`r`, :math:`r_p`, or
        :math:`\theta`. Expected units for distances are :math:`\mathrm{Mpc}/h`
        and degrees for angles. Length of nbins+1
    cosmo : :class:`~nbodykit.cosmology.cosmology.Cosmology`, optional
        the cosmology instance used to convert redshift into comoving distance;
        this is required for all cases except ``mode='angular'``
    second : CatalogSource, optional
        the second source of particles to cross-correlate
    Nmu : int, optional
        the number of :math:`\mu` bins, ranging from 0 to 1; requred if
        ``mode='2d'``
    pimax : float, optional
        The maximum separation along the line-of-sight when ``mode='projected'``.
        Distances along the :math:`\pi` direction are binned with unit
        depth. For instance, if ``pimax=40``, then 40 bins will be created
        along the :math:`\pi` direction.
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
    domain_factor : int, optional
        the integer value by which to oversubscribe the domain decomposition
        mesh before balancing loads; this number can affect the distribution
        of loads on the ranks -- an optimal value will lead to balanced loads
    **config : key/value pairs
        additional keywords to pass to the :mod:`Corrfunc` function

    Notes
    -----
    This class can compute pair counts using several different coordinate
    choices, based on the value of the input argument ``mode``. The choices
    are:

    * ``mode='1d'`` : compute pairs as a function of the 3D separation :math:`r`
    * ``mode='2d'`` : compute pairs as a function of the 3D separation :math:`r`
      and the cosine of the angle to the line-of-sight, :math:`\mu`
    * ``mode='projected'`` : compute pairs as a function of distance perpendicular
      and parallel to the line-of-sight, :math:`r_p` and :math:`\pi`
    * ``mode='angular'`` : compute pairs as a function of angle on the sky, :math:`\theta`
    """
    logger = logging.getLogger('SurveyDataPairCount')

    def __init__(self, mode, first, edges, cosmo=None, second=None,
                    Nmu=None, pimax=None,
                    ra='RA', dec='DEC', redshift='Redshift', weight='Weight',
                    show_progress=False, domain_factor=4,
                    **config):

        # verify the input sources
        required_cols = [ra, dec, weight]
        if mode != 'angular': required_cols.append(redshift)
        verify_input_sources(first, second, None, required_cols, inspect_boxsize=False)

        # init the base class (this verifies input arguments)
        PairCountBase.__init__(self, mode, edges, first, second, Nmu, pimax, weight, show_progress)

        # need cosmology if not angular!
        if mode != 'angular' and cosmo is None:
            raise ValueError("'cosmo' keyword is required when 'mode' is not 'angular'")

        # save the meta-data
        self.attrs['cosmo'] = cosmo
        self.attrs['weight'] = weight
        self.attrs['ra'] = ra
        self.attrs['dec'] = dec
        self.attrs['redshift'] = redshift
        self.attrs['config'] = config
        self.attrs['domain_factor'] = domain_factor

        # run the algorithm
        self.run()

    def run(self):
        """
        Calculate the pair counts of a survey data catalog.
        This adds the following attribute:

        - :attr:`SurveyDataPairCount.pairs`

        self.pairs.attrs['total_wnpairs']: The total of wnpairs.

        Attributes
        ----------
        pairs : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object holding the pair count results.
            The coordinate grid will be ``(r,)``, ``(r,mu)``, ``(rp, pi)``,
            or ``(theta,)`` when ``mode`` is '1d', '2d', 'projected', 'angular',
            respectively.

            The BinnedStatistic stores the following variables:

            - ``r``, ``rp``, or ``theta`` : the mean separation value in the bin
            - ``npairs``: the number of pairs in the bin
            - ``wnpairs``: the weighted npairs in the bin; each pair
              contributes the product of the individual weight values


        """
        from .domain import decompose_survey_data

        # setup
        mode = self.attrs['mode']
        first, second = self.first, self.second
        attrs = self.attrs.copy()
        Nmu = 1 if mode == '1d' else attrs['Nmu']

        # compute the max cartesian distance for smoothing
        smoothing = numpy.max(attrs['edges'])
        if mode == 'projected':
            smoothing = numpy.sqrt(smoothing**2 + attrs['pimax']**2)
        elif mode == 'angular':
            smoothing = 2 * numpy.sin(0.5 * numpy.deg2rad(smoothing))

        # do a domain decomposition on the data
        (pos1, w1), (pos2, w2) = decompose_survey_data(first, second, attrs,
                                                        self.logger, smoothing,
                                                        angular=(mode=='angular'),
                                                        domain_factor=attrs['domain_factor'])

        # get the Corrfunc callable based on mode
        if attrs['mode'] in ['1d', '2d']:
            from .corrfunc.mocks import DDsmu_mocks
            func = DDsmu_mocks(attrs['edges'], Nmu, comm=self.comm, show_progress=attrs['show_progress'])

        elif attrs['mode'] == 'projected':
            from .corrfunc.mocks import DDrppi_mocks
            func = DDrppi_mocks(attrs['edges'], attrs['pimax'], comm=self.comm, show_progress=attrs['show_progress'])

        elif attrs['mode'] == 'angular':
            from .corrfunc.mocks import DDtheta_mocks
            func = DDtheta_mocks(attrs['edges'], comm=self.comm, show_progress=attrs['show_progress'])

        # do the calculation
        self.pairs = func(pos1, w1, pos2, w2, **attrs['config'])
        self.pairs.attrs['total_wnpairs'] = self.attrs['total_wnpairs']

        # squeeze the result if '1d' (single mu bin was used)
        if mode == '1d':
            self.pairs = self.pairs.squeeze()
