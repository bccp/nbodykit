import numpy
import logging
from six import string_types

from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.utils import split_size_3d
from pmesh.domain import GridND

try:
    import Corrfunc
    from Corrfunc.mocks import DDsmu_mocks
except:
    Corrfunc = None

class PairCountBase(object):
    """
    Base class for pair counting algorithms, either for a simulation box
    or survey data

    Parameters
    ----------
    mode : {'1d', '2d'}
        compute paircounts as a function of ``r`` and ``mu`` or just ``r``
    source1 : CatalogSource
        the first source of particles
    source2 : CatalogSource, optional
        the second source of particles to cross-correlate
    redges : array_like
        the radius bin edges; length of nbins+1
    Nmu : int
        the number of ``mu`` bins to use; bins range from [0,1]
    """
    def __init__(self, mode, source1, source2, redges, Nmu):

        assert mode in ['1d', '2d'], "PairCount mode must be '1d' or '2d'"
        self.source1 = source1
        self.source2 = source2
        self.comm = source1.comm

        # save the meta-data
        self.attrs = {}
        self.attrs['mode'] = mode
        self.attrs['redges'] = redges
        self.attrs['Nmu'] = Nmu

    def __getstate__(self):
        return {'result':self.result.data, 'attrs':self.attrs}

    def __setstate__(self, state):
        self.__dict__.update(state)
        redges = self.attrs['redges']
        if self.attrs['mode'] == '1d':
            self.result = BinnedStatistic(['r'], [redges], self.result, fields_to_sum=['npairs'])
        else:
            muedges = numpy.linspace(0, 1., self.attrs['Nmu']+1)
            self.result = BinnedStatistic(['r', 'mu'], [redges, muedges], self.result, fields_to_sum=['npairs'])

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


class SimulationBoxPairCount(PairCountBase):
    """
    Count (weighted) pairs of objects in a simulation box using the
    :mod:`Corrfunc` package.

    This uses the :func:`Corrfunc.theory.DD.DD` and
    :func:`Corrfunc.theory.DDsmu.DDsmu` functions to count pairs.

    Results are computed when the object is inititalized. See the documenation
    of :func:`~SimulationBoxPairCount.run` for the attributes storing the
    results.

    .. note::

        The algorithm expects the positions of particles in a simulation box to
        be the Cartesian ``x``, ``y``, and ``z`` vectors. To compute
        pair counts on survey data, using right ascension, declination, and
        redshift, see :class:`SurveyDataPairCount`.

    Parameters
    ----------
    mode : {'1d', '2d'}
        compute pair counts as a function of ``r`` and ``mu`` or just ``r``
    source1 : CatalogSource
        the first source of particles, providing the 'Position' column
    redges : array_like
        the radius bin edges; length of nbins+1
    BoxSize : float, 3-vector, optional
        the size of the box; if 'BoxSize' is not provided in the source
        'attrs', it must be provided here
    source2 : CatalogSource, optional
        the second source of particles to cross-correlate
    Nmu : int, optional
        the number of ``mu`` bins, ranging from 0 to 1
    los : {'x', 'y', 'z'}, int, optional
        the axis of the simulation box to treat as the line-of-sight direction;
        this can be provided as string identifying one of 'x', 'y', 'z' or
        the equivalent integer number of the axis
    periodic : bool, optional
        whether to use periodic boundary conditions
    weight : str, optional
        the name of the column in the source specifying the particle weights
    **config : key/value pairs
        additional keywords to pass to the :mod:`Corrfunc` function
    """
    logger = logging.getLogger('SimulationBoxPairCount')

    def __init__(self, mode, source1, redges, BoxSize=None, periodic=True,
                    source2=None, Nmu=5, los='z', weight='Weight', **config):

        if isinstance(los, string_types):
            assert los in 'xyz', "``los`` should be one of 'x', 'y', 'z'"
            los = 'xyz'.index(los)
        elif isinstance(los, int):
            if los < 0: los += 3
            assert los in [0,1,2], "``los`` should be one of 0, 1, 2"
        else:
            raise ValueError("``los`` should be either ['x', 'y', 'z'] or [0,1,2]")

        # verify the input sources
        BoxSize = verify_input_sources(source1, source2, BoxSize, ['Position', weight])

        # init the base class
        PairCountBase.__init__(self, mode, source1, source2, redges, Nmu)

        # save the meta-data
        self.attrs['BoxSize'] = BoxSize
        self.attrs['periodic'] = periodic
        self.attrs['weight'] = weight
        self.attrs['config'] = config
        self.attrs['los'] = los

        # test Rmax for PBC
        if periodic and numpy.amax(redges) > 0.5*self.attrs['BoxSize'].min():
            raise ValueError("periodic pair counts cannot be computed for Rmax > 0.5 * BoxSize")

        # run the algorithm
        self.run()

    def run(self):
        """
        Calculate the 3D pair-counts in a simulation box as a function
        of separation ``r`` or separation and angle to line-of-sight
        (``r``, ``mu``). This adds the following attributes to the class:

        - :attr:`SimulationBoxPairCount.result`

        Attributes
        ----------
        result : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object holding the pair count and correlation
            function results. The coordinate grid is either ``r`` or
            ``r`` and ``mu``. It stores the following variables:

            - ``r``: the mean separation value in the bin
            - ``xi``: the mean correlation function value in the bin, computed as
              :math:`DD/RR - 1`, where :math:`RR` is the number of random pairs
              in the bin
            - ``npairs``: the number of pairs in the bin
            - ``weightavg``: the average weight value in the bin; each pair
              contributes the product of the individual weight values
        """
        if Corrfunc is None:
            raise ImportError(("please install Corrfunc using either ``conda install -c bccp corrfunc``"
                               " or from ``pip install pip install git+git://github.com/nickhand/Corrfunc``"))


        # some setup
        redges = self.attrs['redges']
        comm   = self.comm
        nbins  = len(redges)-1
        boxsize = self.attrs['BoxSize']
        L = None

        # determine axes perp and paralell to
        los = self.attrs['los']
        perp_axes = [0,1,2]
        perp_axes.remove(los)

        # cubic box required by Corrunc
        if self.attrs['periodic']:
            if numpy.all(boxsize == boxsize[0]):
                L = boxsize[0]
            else:
                raise ValueError(("``Corrfunc`` does not currently support periodic "
                                  "wrapping with non-cubic boxes"))

        # determine processor division for domain decomposition
        np = split_size_3d(comm.size)
        if self.comm.rank == 0:
            self.logger.info("using cpu grid decomposition: %s" %str(np))

        # get the (periodic-enforced) position
        pos1 = self.source1['Position']
        if self.attrs['periodic']:
            pos1 %= boxsize
        pos1, w1 = self.source1.compute(pos1, self.source1[self.attrs['weight']])
        N1 = comm.allreduce(len(pos1))

        if self.source2 is not None:
            pos2 = self.source2['Position']
            if self.attrs['periodic']:
                pos2 %= boxsize
            pos2, w2 = self.source2.compute(pos2, self.source2[self.attrs['weight']])
            N2 = comm.allreduce(len(pos2))
        else:
            pos2 = pos1
            w2 = w1
            N2 = N1

        # global min/max across all ranks
        posmin = numpy.asarray(comm.allgather(pos1.min(axis=0))).min(axis=0)
        posmax = numpy.asarray(comm.allgather(pos1.max(axis=0))).max(axis=0)

        # domain decomposition
        grid = [
            numpy.linspace(0, boxsize[0], np[0] + 1, endpoint=True),
            numpy.linspace(0, boxsize[1], np[1] + 1, endpoint=True),
            numpy.linspace(0, boxsize[2], np[2] + 1, endpoint=True),
        ]
        domain = GridND(grid, comm=comm)

        layout = domain.decompose(pos1, smoothing=0)
        pos1   = layout.exchange(pos1)
        w1     = layout.exchange(w1)

        # get the position/weight of the secondaries
        rmax = numpy.max(redges)
        if rmax > boxsize.max() * 0.25:
            pos2 = numpy.concatenate(comm.allgather(pos2), axis=0)
            w2   = numpy.concatenate(comm.allgather(w2), axis=0)
        else:
            layout  = domain.decompose(pos2, smoothing=rmax)
            pos2 = layout.exchange(pos2)
            w2   = layout.exchange(w2)

        # log the sizes of the trees
        self.logger.info('rank %d correlating %d x %d' %(comm.rank, len(pos1), len(pos2)))
        if comm.rank == 0: self.logger.info('all correlating %d x %d' %(N1, N2))

        # do the pair counting
        kws = {}
        kws['weights1'] = w1.astype(pos1.dtype)
        kws['periodic'] = self.attrs['periodic']
        kws['X1'] = pos1[:,perp_axes[0]]
        kws['Y1'] = pos1[:,perp_axes[1]]
        kws['Z1'] = pos1[:,los] # LOS defined with respect to this axis
        kws['X2'] = pos2[:,perp_axes[0]]
        kws['Y2'] = pos2[:,perp_axes[1]]
        kws['Z2'] = pos2[:,los]
        kws['weights2'] = w2.astype(pos2.dtype)
        kws['weight_type'] = 'pair_product'
        if L is not None: kws['boxsize'] = L

        # 1D or 2D calculations
        if self.attrs['mode'] == '1d':
            kws['output_ravg'] = True
            kws.update(self.attrs['config'])
            try:
                pc = Corrfunc.theory.DD(0, 1, redges, **kws)
            except:
                raise RuntimeError("error when calling Corrfunc.theory.DD function")
            rcol = 'ravg'
        else:
            kws['output_savg'] = True
            kws.update(self.attrs['config'])
            try:
                pc = Corrfunc.theory.DDsmu(0, 1, redges, 1.0, self.attrs['Nmu'], **kws)
            except:
                raise RuntimeError("error when calling Corrfunc.theory.DDsmu function")
            pc = pc.reshape((-1, self.attrs['Nmu']))
            rcol = 'savg'

        self.logger.info('...rank %d done correlating' %(comm.rank))

        # make a new structured array
        dtype = numpy.dtype([('r', 'f8'), ('xi', 'f8'), ('npairs', 'f8'), ('weightavg', 'f8')])
        data = numpy.empty(pc.shape, dtype=dtype)

        # do the sum across ranks
        data['r'][:] = comm.allreduce(pc['npairs']*pc[rcol])
        data['weightavg'][:] = comm.allreduce(pc['npairs']*pc['weightavg'])
        data['npairs'][:] = comm.allreduce(pc['npairs'])
        idx = data['npairs'] > 0.
        data['r'][idx] /= data['npairs'][idx]
        data['weightavg'][idx] /= data['npairs'][idx]

        # compute the random pairs from the fractional volume
        RR = 1.*N1*N2 / boxsize.prod()
        if self.attrs['mode'] == '2d':
            dr3 = numpy.diff(redges**3)
            muedges = numpy.linspace(0, 1, self.attrs['Nmu']+1)
            dmu = numpy.diff(muedges)
            RR *= 2. / 3. * numpy.pi * dr3[:,None] * dmu[None,:]
        else:
            RR *= 4. / 3. * numpy.pi * numpy.diff(redges**3)

        # correlation function value
        data['xi'] = (1. * data['npairs'] / RR) - 1.0

        # make the BinnedStatistic
        if self.attrs['mode'] == '1d':
            self.result = BinnedStatistic(['r'], [redges], data, fields_to_sum=['npairs'])
        else:
            self.result = BinnedStatistic(['r','mu'], [redges,muedges], data, fields_to_sum=['npairs'])

class SurveyDataPairCount(PairCountBase):
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
    source1 : CatalogSource
        the first source of particles, providing the 'Position' column
    redges : array_like
        the radius bin edges; length of nbins+1
    cosmo : :class:`~nbodykit.cosmology.core.Cosmology`
        the cosmology instance used to convert redshift into comoving distance
    source2 : CatalogSource, optional
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
    **config : key/value pairs
        additional keywords to pass to the :mod:`Corrfunc` function
    """
    logger = logging.getLogger('SurveyDataPairCount')

    def __init__(self, mode, source1, redges, cosmo, source2=None, Nmu=5,
                    ra='RA', dec='DEC', redshift='Redshift', weight='Weight', **config):

        # verify the input sources
        verify_input_sources(source1, source2, None, [ra, dec, redshift, weight], inspect_boxsize=False)

        # init the base class
        PairCountBase.__init__(self, mode, source1, source2, redges, Nmu)

        # save the meta-data
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
            a BinnedStatistic object holding the pair count and correlation
            function results. The coordinate grid is either ``r`` or
            ``r`` and ``mu``. It stores the following variables:

            - ``r``: the mean separation value in the bin
            - ``npairs``: the number of pairs in the bin
            - ``weightavg``: the average weight value in the bin; each pair
              contributes the product of the individual weight values
        """
        if Corrfunc is None:
            raise ImportError(("please install Corrfunc using either ``conda install -c bccp corrfunc``"
                               " or from ``pip install pip install git+git://github.com/nickhand/Corrfunc``"))
        from nbodykit.transform import StackColumns

        # some setup
        redges = self.attrs['redges']
        comm   = self.comm
        nbins  = len(redges)-1
        poscols = [self.attrs['ra'], self.attrs['dec'], self.attrs['redshift']]
        Nmu = 1 if self.attrs['mode'] == '1d' else self.attrs['Nmu']

        # determine processor division for domain decomposition
        np = split_size_3d(comm.size)
        if self.comm.rank == 0:
            self.logger.info("using cpu grid decomposition: %s" %str(np))

        # get the (periodic-enforced) position
        pos1 = StackColumns(*[self.source1[col] for col in poscols]) # this is RA, DEC, REDSHIFT
        pos1, w1 = self.source1.compute(pos1, self.source1[self.attrs['weight']])
        rdist1 = get_comoving_dist(comm, pos1, self.attrs['cosmo'])
        pos1[:,2] = rdist1
        N1 = comm.allreduce(len(pos1))

        if self.source2 is not None:
            pos2 = StackColumns(*[self.source2[col] for col in poscols])
            pos2, w2 = self.source2.compute(pos2, self.source2[self.attrs['weight']])
            rdist2 = get_comoving_dist(comm, pos2, self.attrs['cosmo'])
            pos2[:,2] = rdist2
            N2 = comm.allreduce(len(pos2))
        else:
            pos2 = pos1
            w2 = w1
            N2 = N1

        # the source2 particles
        pos2 = numpy.concatenate(comm.allgather(pos2), axis=0)
        w2   = numpy.concatenate(comm.allgather(w2), axis=0)

        # log the sizes of the trees
        self.logger.info('rank %d correlating %d x %d' %(comm.rank, len(pos1), len(pos2)))
        if comm.rank == 0: self.logger.info('all correlating %d x %d' %(N1, N2))

        if not len(pos1) or not len(pos2):
            dtype = [('savg', 'f8'), ('npairs', 'u8'), ('weightavg', 'f8')]
            pc = numpy.zeros((nbins, Nmu), dtype=dtype)
        else:
            # do the pair counting
            kws = {}
            kws['weights1'] = w1.astype(pos1.dtype)
            kws['is_comoving_dist'] = True
            kws['RA1'] = pos1[:,0] # ra
            kws['DEC1'] = pos1[:,1] # dec
            kws['CZ1'] = pos1[:,2] # comoving distance
            kws['RA2'] = pos2[:,0]
            kws['DEC2'] = pos2[:,1]
            kws['CZ2'] = pos2[:,2]
            kws['weights2'] = w2.astype(pos2.dtype)
            kws['weight_type'] = 'pair_product'
            kws['output_savg'] = True
            kws.update(self.attrs['config'])

            try:
                pc = Corrfunc.mocks.DDsmu_mocks(0, 1, 1, Nmu, 1.0, redges, **kws)
            except:
                raise RuntimeError("error when calling Corrfunc.mocks.DDsmu_mocks function")
            pc = pc.reshape((-1, Nmu))

        self.logger.info('...rank %d done correlating' %(comm.rank))

        # make a new structured array
        dtype = numpy.dtype([('r', 'f8'), ('npairs', 'f8'), ('weightavg', 'f8')])
        data = numpy.zeros(pc.shape, dtype=dtype)

        # do the sum across ranks
        data['r'][:] = comm.allreduce(pc['npairs']*pc['savg'])
        data['weightavg'][:] = comm.allreduce(pc['npairs']*pc['weightavg'])
        data['npairs'][:] = comm.allreduce(pc['npairs'])
        idx = data['npairs'] > 0.
        data['r'][idx] /= data['npairs'][idx]
        data['weightavg'][idx] /= data['npairs'][idx]

        # make the BinnedStatistic
        if self.attrs['mode'] == '1d':
            self.result = BinnedStatistic(['r'], [redges], numpy.squeeze(data), fields_to_sum=['npairs'])
        else:
            muedges = numpy.linspace(0, 1., Nmu+1)
            self.result = BinnedStatistic(['r','mu'], [redges,muedges], data, fields_to_sum=['npairs'])

def verify_input_sources(first, second, BoxSize, required_columns, inspect_boxsize=True):
    """
    Verify the input source objects have
    """
    if second is None: second = first

    # check for comm mismatch
    assert second.comm is first.comm, "communicator mismatch between input sources"

    # check required columns
    for source in [first, second]:
        for col in required_columns:
            if col not in source:
                raise ValueError("the column '%s' is missing from input source; cannot do pair count" %col)

    if inspect_boxsize:
        _BoxSize = numpy.zeros(3)
        BoxSize1 = first.attrs.get('BoxSize', None)
        if BoxSize1 is not None:
            _BoxSize[:] = BoxSize1[:]
        if BoxSize is not None:
            _BoxSize[:] = BoxSize
        if (_BoxSize==0.).all():
            raise ValueError("BoxSize must be supplied in the source ``attr`` or via the ``BoxSize`` keyword")

        BoxSize2 = second.attrs.get('BoxSize', None)
        if BoxSize1 is not None and BoxSize2 is not None:
            if not numpy.array_equal(first.attrs['BoxSize'], BoxSize2):
                raise ValueError("BoxSize mismatch between pair count cross-correlation sources")
            if not numpy.array_equal(first.attrs['BoxSize'], _BoxSize):
                raise ValueError("BoxSize mismatch between sources and the pair count algorithm")

        return _BoxSize

def get_comoving_dist(comm, pos, cosmo):
    """
    Return comoving distances from RA, DEC, Redshift

    ``pos`` has 3 columns giving: ra, dec, redshift
    """
    ra, dec, redshift = numpy.deg2rad(pos[:,0]), numpy.deg2rad(pos[:,1]), pos[:,2]

    # compute comoving distance
    zmin, zmax = redshift.min(), redshift.max()
    zbins = numpy.logspace(numpy.log10(zmin), numpy.log10(zmax), 512)
    comoving_distance = cosmo.comoving_distance.fit('z', bins=zbins)
    rdist = comoving_distance(redshift).value * cosmo.h

    return rdist
