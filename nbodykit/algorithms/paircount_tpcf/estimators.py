from nbodykit.binned_statistic import BinnedStatistic
import numpy
import warnings

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
    def __init__(self, mode, edges, BoxSize):

        assert mode in ['1d', '2d', 'projected', 'angular']
        self.mode = mode
        self.edges = edges
        self.BoxSize = BoxSize

    @property
    def filling_factor(self):
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
        Evaluate the expected randoms pair counts.
        """
        if NR2 is None:
            return NR1 ** 2  * self.filling_factor
        else:
            return NR1 * NR2 * self.filling_factor

def LandySzalayEstimator(pair_counter, data1, data2, randoms1, randoms2, logger=None, **kwargs):
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

    # data1 x data2
    if logger is not None and comm.rank == 0:
        logger.info("computing data1 - data2 pair counts")
    D1D2 = pair_counter(first=data1, second=data2, **kwargs).pairs

    if randoms2 is None:
        randoms2 = randoms1

    # determine the sample sizes
    ND1, NR1 = data1.csize, randoms1.csize
    ND2 = data2.csize if data2 is not None else ND1
    NR2 = randoms2.csize

    # do data - randoms correlation
    if logger is not None and comm.rank == 0:
        logger.info("computing data1 - randoms2 pair counts")
    D1R2 = pair_counter(first=data1, second=randoms2, **kwargs).pairs

    if data2 is not None:
        if logger is not None and comm.rank == 0:
            logger.info("computing data2 - randoms1 pair counts")
        D2R1 = pair_counter(first=data2, second=randoms1, **kwargs).pairs
    else:
        D2R1 = D1R2

    # and randoms - randoms calculation
    if logger is not None and comm.rank == 0:
        logger.info("computing randoms1 - randoms2 pair counts")
    R1R2 = pair_counter(first=randoms1, second=randoms2, **kwargs).pairs

    # init
    CF = numpy.zeros(D1D2.shape)
    CF[:] = numpy.nan

    fN1 = float(NR1)/ND1
    fN2 = float(NR2)/ND2
    nonzero = R1R2['npairs'] > 0

    # the Landy - Szalay estimator
    # (DD - DR - RD + RR) / RR
    xi = fN1 * fN2 * (D1D2['npairs']*D1D2['weightavg'])[nonzero]
    xi -= fN1 * (D1R2['npairs']*D1R2['weightavg'])[nonzero]
    xi -= fN2 * (D2R1['npairs']*D2R1['weightavg'])[nonzero]
    xi /= (R1R2['npairs']*R1R2['weightavg'])[nonzero]
    xi += 1.
    CF[nonzero] = xi[:]

    # warn about NaNs in the estimator
    if data1.comm.rank == 0 and numpy.isnan(CF).any():
        msg = ("The RR calculation in the Landy-Szalay estimator contains"
        " separation bins with no bins. This will result in NaN values in the resulting"
        " correlation function. Try increasing the number of randoms and/or using"
        " broader bins.")
        warnings.warn(msg)

    CF = _create_tpcf_result(D1D2, CF)
    return D1D2, D1R2, D2R1, R1R2, CF


def NaturalEstimator(data_paircount):
    """
    Internal function to computing the correlation function using
    analytic randoms and the so-called "natural" correlation function
    estimator, :math:`DD/RR - 1`.
    """
    # data1 x data2
    D1D2 = data_paircount.pairs
    attrs = data_paircount.attrs

    # determine the sample sizes
    ND1, ND2 = attrs['N1'], attrs['N2']
    edges = D1D2.edges
    mode = attrs['mode']
    BoxSize = attrs['BoxSize']

    # analytic randoms - randoms calculation assuming uniform distribution
    _R1R2 = AnalyticUniformRandoms(mode, edges, BoxSize)(ND1, ND2)
    edges = [D1D2.edges[d] for d in D1D2.dims]
    R1R2 = BinnedStatistic(D1D2.dims, edges, _R1R2.view([('npairs', 'f8')]))

    # and compute the correlation function as DD/RR - 1
    CF = (D1D2['npairs']*D1D2['weightavg']) / R1R2['npairs'] - 1.

    # create a BinnedStatistic holding the CF
    CF = _create_tpcf_result(D1D2, CF)

    return R1R2, CF

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
    return BinnedStatistic(D1D2.dims, edges, data)
