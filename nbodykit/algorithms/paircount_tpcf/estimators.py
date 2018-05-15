from nbodykit.binned_statistic import BinnedStatistic
import numpy
import warnings

class WedgeBinnedStatistic(BinnedStatistic):
    """
        A BinnedStatistic of wedges that can be converted to multiples.
    """

    def to_poles(self, poles):
        r"""
        Invert the measured wedges :math:`\xi(r,mu)` into correlation
        multipoles, :math:`\xi_\ell(r)`.

        To select a mu_range, use

        .. code::

            poles = self.sel(mu=slice(*mu_range), method='nearest').to_poles(poles)

        Parameters
        ----------
        poles: array_like
            the list of multipoles to compute

        Returns
        -------
        xi_ell : BinnedStatistic
            a data set holding the :math:`\xi_\ell(r)` multipoles
        """
        from scipy.special import legendre
        from scipy.integrate import quad

        # new data array
        x = str(self.dims[0])
        dtype = numpy.dtype([(x, 'f8')] + [('corr_%d' %ell, 'f8') for ell in poles])
        data = numpy.zeros((self.shape[0]), dtype=dtype)
        dims = [x]
        edges = [self.edges[x]]

        # FIXME: use something fancier than the central point.
        mu_bins = numpy.diff(self.edges['mu'])
        mu_mid = (self.edges['mu'][1:] + self.edges['mu'][:-1])/2.

        for ell in poles:
            legendrePolynomial = (2.*ell+1.)*legendre(ell)(mu_mid)
            data['corr_%d' %ell] = numpy.sum(self['corr']*legendrePolynomial*mu_bins,axis=-1)/numpy.sum(mu_bins)

        data[x] = numpy.mean(self[x],axis=-1)

        return BinnedStatistic(dims=dims, edges=edges, data=data, poles=poles)


class AnalyticUniformRandoms(object):
    """
    Internal class to compute analytic pair counts for uniformly
    distributed randoms.

    This can compute the expected randoms counts for several coordinate
    choices, based on ``mode``. The relevant volume/area calculations for the
    various coordinate modes are:

    * mode='1d': volume of sphere
    * mode='2d': volume of spherical sector
    * mode='projected': volume of cylinder
    * mode='angular': area of spherical cap
    """
    def __init__(self, mode, dims, edges, BoxSize):

        assert mode in ['1d', '2d', 'projected', 'angular']
        self.mode = mode
        self.edges = edges
        self.dims = dims
        self.BoxSize = BoxSize

    def get_filling_factor(self):
        """
        This gives the ratio of the volume (or area) occupied by each bin to
        the global volume (or area).

        It is different based on the value of :attr:`mode`.
        """
        # based on a volume of a sphere
        if self.mode == '1d':
            r_edges = self.edges['r']
            V = 4. / 3. * numpy.pi * r_edges**3
            dV = numpy.diff(V)
            return dV / self.BoxSize.prod()

        # based on the volume of a spherical sector
        elif self.mode == '2d':
            r_edges, mu_edges = self.edges['r'], self.edges['mu']
            V = 2. / 3. * numpy.pi * numpy.outer(r_edges**3, mu_edges)
            V *= 2.0 # accounts for the fact that we bin in abs(mu)
            dV = numpy.diff(numpy.diff(V, axis=0), axis=1)
            return  dV / self.BoxSize.prod()

        # based on volume of a cylinder
        elif self.mode == 'projected':
            rp_edges, pi_edges = self.edges['rp'], self.edges['pi']
            V = numpy.pi * numpy.outer(rp_edges**2, (2. * pi_edges)) # height is double pimax!
            dV = numpy.diff(numpy.diff(V, axis=0), axis=1)
            return  dV / self.BoxSize.prod()

        # based on the surface area of a spherical cap
        elif self.mode == 'angular':
            theta_bins = self.edges['theta']
            chord = 2*numpy.sin(0.5*numpy.deg2rad(theta_bins)) # chord distance
            h = 1. - numpy.sqrt(1. - chord**2)
            A = numpy.pi*(chord**2 + h**2)
            dA = numpy.diff(A)
            return dA / (4*numpy.pi)

    def __call__(self, NR1, NR2=None):
        """
        Evaluate the expected randoms pair counts, and the total_weight, returns
        as an object that looks like the result of paircount.

        """
        edges = [self.edges[d] for d in self.dims] # sequentialize it, poor API!

        if NR2 is None:
            R1R2 = NR1 ** 2  * self.get_filling_factor()
            total_wnpairs = NR1 * (NR1 - 1) * 0.5
        else:
            R1R2 = NR1 * NR2 * self.get_filling_factor()
            total_wnpairs = NR1 * NR2 * 0.5

        data = numpy.empty_like(R1R2, dtype=[('npairs', 'f8'), ('wnpairs', 'f8')])
        data['npairs'] = R1R2
        data['wnpairs'] = R1R2
        pairs = WedgeBinnedStatistic(self.dims, edges, data)

        R1R2 = lambda : None
        R1R2.attrs = {}
        R1R2.pairs = pairs
        R1R2.attrs['total_wnpairs']= total_wnpairs
        R1R2.pairs.attrs['total_wnpairs']= total_wnpairs

        return R1R2

def LandySzalayEstimator(pair_counter, data1, data2, randoms1, randoms2, R1R2=None, logger=None, **kwargs):
    """
    Compute the correlation function from data/randoms using the
    Landy - Szalay estimator to compute the correlation function.

    Parameters
    ----------
    pair_counter : SimulationBoxPairCount, SurveyDataPairCount
        the pair counting algorithm class
    data1 : CatalogSource
        the first data source
    data2 : CatalogSource, None
        the second data catalog to cross-correlate; can be None for auto-correlations
    randoms1 : CatalogSource
        the randoms catalog corresponding to ``data1``
    randoms2 : CatalogSource, None
        the second randoms catalog; can be None for auto-correlations
    R1R2 : SimulationBoxPairCount, SurveyDataPairCount, optional
        if provided, random pairs R1R2 are not recalculated
    **kwargs :
        the parameters passed to the ``pair_counter`` class to count pairs

    Returns
    -------
    D1D2, D1R2, D2R1, R1R2, CF : BinnedStatistic
        the various terms of the LS estimator + the correlation function result

    References
    ----------
    http://adsabs.harvard.edu/abs/1993ApJ...412...64L
    """
    # make sure we have the randoms
    assert randoms1 is not None
    comm = data1.comm

    if randoms2 is None: randoms2 = randoms1

    # and randoms - randoms calculation
    if logger is not None and comm.rank == 0:
        logger.info("computing randoms1 - randoms2 pair counts")
    if not R1R2:
        R1R2 = pair_counter(first=randoms1, second=randoms2, **kwargs)

    # data1 x data2
    if logger is not None and comm.rank == 0:
        logger.info("computing data1 - data2 pair counts")
    D1D2 = pair_counter(first=data1, second=data2, **kwargs)

    # do data - randoms correlation
    if logger is not None and comm.rank == 0:
        logger.info("computing data1 - randoms2 pair counts")
    D1R2 = pair_counter(first=data1, second=randoms2, **kwargs)

    if data2 is not None:
        if logger is not None and comm.rank == 0:
            logger.info("computing data2 - randoms1 pair counts")
        D2R1 = pair_counter(first=data2, second=randoms1, **kwargs)
    else:
        D2R1 = D1R2

    fDD = R1R2.attrs['total_wnpairs']/D1D2.attrs['total_wnpairs']
    fDR = R1R2.attrs['total_wnpairs']/D1R2.attrs['total_wnpairs']
    fRD = R1R2.attrs['total_wnpairs']/D2R1.attrs['total_wnpairs']

    nonzero = R1R2.pairs['npairs'] > 0

    # init
    CF = numpy.zeros(D1D2.pairs.shape)
    CF[:] = numpy.nan
    Error = numpy.zeros(D1D2.pairs.shape)
    Error[:] = numpy.nan

    # the Landy - Szalay estimator
    # (DD - DR - RD + RR) / RR
    DD = (D1D2.pairs['wnpairs'])[nonzero]
    DR = (D1R2.pairs['wnpairs'])[nonzero]
    RD = (D2R1.pairs['wnpairs'])[nonzero]
    RR = (R1R2.pairs['wnpairs'])[nonzero]
    xi = (fDD * DD - fDR * DR - fRD * RD)/RR + 1
    CF[nonzero] = xi[:]

    # warn about NaNs in the estimator
    if comm.rank == 0 and numpy.isnan(CF).any():
        msg = ("The RR calculation in the Landy-Szalay estimator contains"
        " separation bins with no bins. This will result in NaN values in the resulting"
        " correlation function. Try increasing the number of randoms and/or using"
        " broader bins.")
        warnings.warn(msg)

    CF = _create_tpcf_result(D1D2.pairs, CF)
    return D1D2.pairs, D1R2.pairs, D2R1.pairs, R1R2.pairs, CF

def NaturalEstimator(D1D2):
    """
    Internal function to computing the correlation function using
    analytic randoms and the so-called "natural" correlation function
    estimator, :math:`DD/RR - 1`.
    """
    attrs = D1D2.attrs

    # determine the sample sizes
    if attrs['is_cross']:
        ND1, ND2 = attrs['N1'], attrs['N2']
    else:
        ND1, ND2 = attrs['N1'], None

    mode = attrs['mode']
    BoxSize = attrs['BoxSize']

    # analytic randoms - randoms calculation assuming uniform distribution
    R1R2 = AnalyticUniformRandoms(mode, D1D2.pairs.dims, D1D2.pairs.edges, BoxSize)(ND1, ND2)

    # and compute the correlation function as DD/RR - 1
    fDD = R1R2.attrs['total_wnpairs'] / D1D2.attrs['total_wnpairs']
    RR = R1R2.pairs['wnpairs']
    DD = D1D2.pairs['wnpairs']

    CF = (DD * fDD) / RR - 1.

    # create a BinnedStatistic holding the CF
    CF = _create_tpcf_result(D1D2.pairs, CF)

    return R1R2.pairs, CF

def _create_tpcf_result(D1D2, CF):
    """
    Create a BinnedStatistic holding the correlation function
    and average bin separation.
    """
    x = D1D2.dims[0]
    data = numpy.empty_like(CF, dtype=[('corr', 'f8'), (x, 'f8')])
    data['corr'] = CF[:]
    data[x] = D1D2[x]
    edges = [D1D2.edges[d] for d in D1D2.dims]
    return WedgeBinnedStatistic(D1D2.dims, edges, data)
