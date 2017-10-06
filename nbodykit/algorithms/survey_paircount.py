from .sim_paircount import PairCountBase, verify_input_sources, MissingCorrfuncError
from nbodykit.binned_statistic import BinnedStatistic

import logging
import numpy

class SurveyPairCountBase(PairCountBase):
    """
    An abstract base class for pair count algorithms from surveys. The input
    data is assumed to be of the form (ra, dec, redshift) for these algorithms.
    """
    def _decompose(self, rmax, angular=False):
        """
        An internal function to perform a domain decomposition on the two
        sources we wish to correlate, returning the position/weights for each
        object in the correlating pair.

        The domain decomposition is based on the Cartesian coordinates of
        the input data (assumed to be in sky coordinates).

        The implementation follows:

        1. Decompose the first source and balance the particle load, such that
           the first source is evenly distributed across all ranks and the
           objects are spatially tight on a given rank.
        2. Decompose the second source, ensuring a given rank holds all
           particles within the desired maximum separation.

        Parameters
        ----------
        rmax : float
            the maximum Cartesian separation implied by the user's binning
        angular : bool, optional
            if ``True``, the Cartesian positions used in the domain
            decomposition are on the unit sphere

        Returns
        -------
        (pos1, w1), (pos2, w2) : array_like
            the (decomposed) set of positions and weights to correlate
        """
        from nbodykit.transform import StackColumns
        from pmesh.domain import GridND
        from nbodykit.utils import split_size_3d

        # either (ra,dec) or (ra,dec,redshift)
        poscols = [self.attrs['ra'], self.attrs['dec']]
        if not angular: poscols += [self.attrs['redshift']]

        # determine processor division for domain decomposition
        np = split_size_3d(self.comm.size)
        if self.comm.rank == 0:
            self.logger.info("using cpu grid decomposition: %s" %str(np))

        # stack position and compute
        pos1 = StackColumns(*[self.first[col] for col in poscols])
        pos1, w1 = self.first.compute(pos1, self.first[self.attrs['weight']])
        N1 = self.comm.allreduce(len(pos1))

        # get comoving dist and boxsize for first
        cosmo = self.attrs.get('cosmo', None)
        cpos1, boxsize1, rdist1 = get_cartesian(self.comm, pos1, cosmo=cosmo)

        # pass in comoving dist to Corrfunc instead of redshift
        if not angular:
            pos1[:,2] = rdist1

        # set up position for second too
        if self.second is not None:

            # stack position and compute for "second"
            pos2 = StackColumns(*[self.second[col] for col in poscols])
            pos2, w2 = self.second.compute(pos2, self.second[self.attrs['weight']])
            N2 = self.comm.allreduce(len(pos2))

            # get comoving dist and boxsize
            cpos2, boxsize2, rdist2 = get_cartesian(self.comm, pos2, cosmo=cosmo)

            # pass in comoving distance instead of redshift
            if not angular:
                pos2[:,2] = rdist2
        else:
            pos2 = pos1
            w2 = w1
            N2 = N1
            boxsize2 = boxsize1
            cpos2 = cpos1

        # determine global boxsize
        if self.second is None:
            boxsize = boxsize1
        else:
            boxsizes = numpy.vstack([boxsize1, boxsize2])
            argmax = numpy.argmax(boxsizes, axis=0)
            boxsize = boxsizes[argmax, [0,1,2]]

        # initialize the domain
        # NOTE: over-decompose by factor of 2 to trigger load balancing
        grid = [
            numpy.linspace(0, boxsize[0], 2*np[0] + 1, endpoint=True),
            numpy.linspace(0, boxsize[1], 2*np[1] + 1, endpoint=True),
            numpy.linspace(0, boxsize[2], 2*np[2] + 1, endpoint=True),
        ]
        domain = GridND(grid, comm=self.comm)

        # balance the load
        domain.loadbalance(domain.load(cpos1))

        # decompose based on cartesian positions
        layout = domain.decompose(cpos1, smoothing=0)
        pos1   = layout.exchange(pos1)
        w1     = layout.exchange(w1)

        # get the position/weight of the secondaries
        if rmax > boxsize.max() * 0.25:
            pos2 = numpy.concatenate(self.comm.allgather(pos2), axis=0)
            w2   = numpy.concatenate(self.comm.allgather(w2), axis=0)
        else:
            layout  = domain.decompose(cpos2, smoothing=rmax)
            pos2 = layout.exchange(pos2)
            w2   = layout.exchange(w2)

        # log the sizes of the trees
        self._log(N1, N2, pos1, pos2)

        return (pos1, w1), (pos2, w2)

class SurveyDataPairCount(SurveyPairCountBase):
    """
    Count (weighted) pairs of objects from a survey data catalog using the
    :mod:`Corrfunc` package.

    This uses the:func:`Corrfunc.mocks.DDsmu_mocks.DDsmu_mocks`
    function to count pairs.

    Results are computed when the object is inititalized. See the documenation
    of :func:`~SurveyDataPairCount.run` for the attributes storing the
    results.

    .. note::

        The algorithm expects the positions of particles from a survey catalog
        be the sky coordinates, right ascension and declination, and redshift.
        To compute pair counts in a simulation box, using the Cartesian
        coordinate vectors, see :class:`SimulationBoxPairCount`.

    .. warning::
        The right ascension and declination columns should be specified
        in degrees.

    Parameters
    ----------
    mode : {'1d', '2d'}
        compute paircounts as a function of ``r`` and ``mu`` or just ``r``
    first : CatalogSource
        the first source of particles, providing the 'Position' column
    redges : array_like
        the radius bin edges; length of nbins+1
    cosmo : :class:`~nbodykit.cosmology.core.Cosmology`
        the cosmology instance used to convert redshift into comoving distance
    second : CatalogSource, optional
        the second source of particles to cross-correlate
    Nmu : int, optional
        the number of ``mu`` bins, ranging from 0 to 1
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
    """
    logger = logging.getLogger('SurveyDataPairCount')

    def __init__(self, mode, first, redges, cosmo,
                    second=None, Nmu=5, ra='RA', dec='DEC',
                    redshift='Redshift', weight='Weight', show_progress=True,
                    **config):

        # check input 'mode'
        assert mode in ['1d', '2d'], "PairCount mode must be '1d' or '2d'"

        # check rmin
        if numpy.min(redges) <= 0.:
            raise ValueError(("the lower edge of the 1st separation bin must "
                              "greater than zero (no self-pairs)"))

        # verify the input sources
        required_cols = [ra, dec, redshift, weight]
        verify_input_sources(first, second, None, required_cols, inspect_boxsize=False)

        # init the base class
        SurveyPairCountBase.__init__(self, first, second, show_progress)

        # save the meta-data
        self.attrs['mode'] = mode
        self.attrs['redges'] = redges
        self.attrs['Nmu'] = Nmu
        self.attrs['cosmo'] = cosmo
        self.attrs['weight'] = weight
        self.attrs['ra'] = ra
        self.attrs['dec'] = dec
        self.attrs['redshift'] = redshift
        self.attrs['config'] = config

        # run the algorithm
        self.run()

    def run(self):
        """
        Calculate the 3D pair-counts of a survey data catalog as a function
        of separation ``r`` or separation and angle to line-of-sight
        (``r``, ``mu``). This adds the following attribute:

        - :attr:`SurveyDataPairCount.result`

        Attributes
        ----------
        result : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object holding the pair count results. The
            coordinate grid is either ``r`` or ``r`` and ``mu``.
            It stores the following variables:

            - ``r``: the mean separation value in the bin
            - ``npairs``: the number of pairs in the bin
            - ``weightavg``: the average weight value in the bin; each pair
              contributes the product of the individual weight values
        """
        try:
            from Corrfunc.mocks import DDsmu_mocks
        except ImportError:
            raise MissingCorrfuncError()

        # some setup
        redges = self.attrs['redges']
        Nmu = 1 if self.attrs['mode'] == '1d' else self.attrs['Nmu']

        # maximum separation value
        rmax = numpy.max(self.attrs['redges'])

        # domain decompose the data
        # NOTE: pos has 3 columns: (ra, dec, redshift)
        (pos1, w1), (pos2, w2) = self._decompose(rmax, angular=False)

        # do the pair counting
        kws = {}
        kws['autocorr'] = 0
        kws['cosmology'] = 1
        kws['nthreads'] = 1
        kws['nmu_bins'] = Nmu
        kws['mu_max'] = 1.0
        kws['binfile'] = self.attrs['redges']
        kws['RA2'] = pos2[:,0]
        kws['DEC2'] = pos2[:,1]
        kws['CZ2'] = pos2[:,2]
        kws['weights2'] = w2.astype(pos2.dtype)
        kws['weight_type'] = 'pair_product'
        kws['output_savg'] = True
        kws['is_comoving_dist'] = True
        kws.update(self.attrs['config'])

        def callback(kws, chunk):
            kws['RA1'] = pos1[chunk][:,0] # ra
            kws['DEC1'] = pos1[chunk][:,1] # dec
            kws['CZ1'] = pos1[chunk][:,2] # comoving distance
            kws['weights1'] = w1[chunk].astype(pos1.dtype)

        # run
        sizes = self.comm.allgather(len(pos1))
        pc = self._run(DDsmu_mocks, kws, sizes, callback=callback)
        pc = pc.reshape((-1, Nmu))

        # sum results over all ranks
        pc = self.comm.allreduce(pc)

        # make a new structured array
        dtype = numpy.dtype([('r', 'f8'), ('npairs', 'u8'), ('weightavg', 'f8')])
        data = numpy.zeros(pc.shape, dtype=dtype)

        # copy over main results
        data['r'] = pc['savg']
        data['npairs'] = pc['npairs']
        data['weightavg'] = pc['weightavg']

        # make the BinnedStatistic
        if self.attrs['mode'] == '1d':
            self.result = BinnedStatistic(['r'], [redges], numpy.squeeze(data), fields_to_sum=['npairs'])
        else:
            muedges = numpy.linspace(0, 1., Nmu+1)
            self.result = BinnedStatistic(['r','mu'], [redges,muedges], data, fields_to_sum=['npairs'])

class AngularPairCount(SurveyPairCountBase):
    """
    Count (weighted) angular pairs of objects from a survey data catalog using the
    :mod:`Corrfunc` package.

    This uses the:func:`Corrfunc.mocks.DDtheta_mocks.DDtheta_mocks`
    function to count pairs.

    Results are computed when the object is inititalized. See the documenation
    of :func:`~AngularPairCount.run` for the attributes storing the
    results.

    .. note::

        The algorithm expects the positions of particles from the survey catalog
        be the right ascension and declination angular coordinates.

    .. warning::
        The right ascension and declination columns should be specified
        in degrees.

    Parameters
    ----------
    first : CatalogSource
        the first source of particles, providing the 'Position' column
    edges : array_like
        the angular separation bin edges (in degrees); length of ``nbins+1``
    second : CatalogSource, optional
        the second source of particles to cross-correlate
    ra : str, optional
        the name of the column in the source specifying the
        right ascension coordinates in units of degrees; default is 'RA'
    dec : str, optional
        the name of the column in the source specifying the declination
        coordinates; default is 'DEC'
    weight : str, optional
        the name of the column in the source specifying the object weights
    show_progress : bool, optional
        if ``True``, perform the pair counting calculation in 10 iterations,
        logging the progress after each iteration; this is useful for
        understanding the scaling of the code
    **config : key/value pairs
        additional keywords to pass to the :mod:`Corrfunc` function
    """
    logger = logging.getLogger('AngularPairCount')

    def __init__(self, first, edges, second=None, ra='RA', dec='DEC',
                    weight='Weight', show_progress=True, **config):

        # check theta min
        if numpy.min(edges) <= 0.:
            raise ValueError(("the lower edge of the 1st separation bin must "
                              "greater than zero (no self-pairs)"))

        # verify the input sources
        verify_input_sources(first, second, None, [ra, dec, weight], inspect_boxsize=False)

        # init the base class
        SurveyPairCountBase.__init__(self, first, second, show_progress)

        # save the meta-data
        self.attrs['edges'] = edges
        self.attrs['ra'] = ra
        self.attrs['dec'] = dec
        self.attrs['weight'] = weight
        self.attrs['config'] = config

        # run the algorithm
        self.run()

    def __setstate__(self, state):
        self.__dict__.update(state)
        edges = self.attrs['edges']
        self.result = BinnedStatistic(['theta'], [edges], self.result, fields_to_sum=['npairs'])

    def run(self):
        """
        Calculate the angular pair counts of a survey data catalog as a function
        of separation angle on the sky. This adds the following attribute:

        - :attr:`AngularPairCount.result`

        Attributes
        ----------
        result : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object holding the pair count results.
            The coordinate grid is ``theta``, the angular separation bins.
            It stores the following variables:

            - ``theta``: the mean separation value in the bin
            - ``npairs``: the number of pairs in the bin
            - ``weightavg``: the average weight value in the bin; each pair
              contributes the product of the individual weight values
        """
        try:
            from Corrfunc.mocks import DDtheta_mocks
        except ImportError:
            raise MissingCorrfuncError()

        # maximum R separation from angular separation
        theta_max = numpy.max(self.attrs['edges'])
        rmax = 2 * numpy.sin(0.5 * numpy.deg2rad(theta_max))

        # domain decompose the data
        # NOTE: pos is angular and only has 2 columns: (ra, dec)
        (pos1, w1), (pos2, w2) = self._decompose(rmax, angular=True)

        # do the pair counting
        kws = {}
        kws['autocorr'] = 0
        kws['nthreads'] = 1
        kws['binfile'] = self.attrs['edges']
        kws['RA2'] = pos2[:,0]
        kws['DEC2'] = pos2[:,1]
        kws['weights2'] = w2.astype(pos2.dtype)
        kws['weight_type'] = 'pair_product'
        kws['output_thetaavg'] = True
        kws.update(self.attrs['config'])

        def callback(kws, chunk):
            kws['RA1'] = pos1[chunk][:,0] # ra
            kws['DEC1'] = pos1[chunk][:,1] # dec
            kws['weights1'] = w1[chunk].astype(pos1.dtype)

        # run
        sizes = self.comm.allgather(len(pos1))
        pc = self._run(DDtheta_mocks, kws, sizes, callback=callback)

        # sum results over all ranks
        pc = self.comm.allreduce(pc)

        # make a new structured array
        dtype = numpy.dtype([('theta', 'f8'), ('npairs', 'u8'), ('weightavg', 'f8')])
        data = numpy.zeros(pc.shape, dtype=dtype)

        # copy over main results
        data['theta'] = pc['thetaavg']
        data['npairs'] = pc['npairs']
        data['weightavg'] = pc['weightavg']

        # make the BinnedStatistic
        edges = [self.attrs['edges']]
        self.result = BinnedStatistic(['theta'], edges, data, fields_to_sum=['npairs'])

def get_cartesian(comm, pos, cosmo=None):
    """
    Convert sky coordinates to Cartesian coordinates and return the implied
    box size from the position bounds.

    If ``cosmo`` is not provided, return coordinates on the unit sphere.
    """
    from nbodykit.utils import get_data_bounds

    # get RA,DEC in degrees
    ra, dec = numpy.deg2rad(pos[:,0]), numpy.deg2rad(pos[:,1])

    # cartesian position
    x = numpy.cos( dec ) * numpy.cos( ra )
    y = numpy.cos( dec ) * numpy.sin( ra )
    z = numpy.sin( dec )
    cpos = numpy.vstack([x,y,z]).T

    # multiply by comoving distance?
    if cosmo is not None:
        assert pos.shape[-1] == 3
        rdist = cosmo.comoving_distance(pos[:,2]) # in Mpc/h
        cpos = rdist[:,None] * cpos
    else:
        rdist = None

    # min/max of position
    cpos_min, cpos_max = get_data_bounds(cpos, comm)
    boxsize = numpy.ceil(abs(cpos_max - cpos_min))

    return cpos, boxsize, rdist
