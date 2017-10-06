import numpy
import logging
from six import string_types

from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic

class MissingCorrfuncError(Exception):
    def __init__(self):
        msg = "use either ``conda install -c bccp corrfunc`` "
        msg += "or ``pip install git+git://github.com/nickhand/Corrfunc``"
        self.args = (msg,)

class CorrfuncResult(object):
    """
    A class used internally holding the array like result of a
    pair counting algorithm from :mod:`Corrfunc`.

    This class is useful for summing pair count results, accounting for
    columns that are pair-weighted.

    Parameters
    ----------
    data : numpy.ndarray
        the numpy structured array result from Corrfunc
    """
    valid = ['weightavg', 'npairs', 'savg', 'ravg', 'thetaavg']

    def __init__(self, data):

        # copy over the valid colums from the input result
        dtype = [(col, data.dtype[col]) for col in data.dtype.names if col in self.valid]
        self.data = numpy.zeros(data.shape, dtype=dtype)
        self.columns = self.data.dtype.names
        for col in self.columns:
            self.data[col] = data[col]

    def __getitem__(self, col):
        return self.data[col]

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def reshape(self, *args, **kwargs):
        self.data = self.data.reshape(*args, **kwargs)
        return self

    def __radd__(self, other):
        return self + other

    def __add__(self, other):

        # all columns are pair-weighted except for "npairs"
        data = numpy.empty_like(self.data)
        for col in self.columns:
            if col == 'npairs': continue
            data[col] = self[col]*self['npairs'] + other[col]*other['npairs']

        # just sum up "npairs"
        data['npairs'] = self['npairs'] + other['npairs']

        # normalize by total "npairs"
        idx = data['npairs'] > 0.
        for col in self.columns:
            if col == 'npairs': continue
            data[col][idx] /= data['npairs'][idx]

        return CorrfuncResult(data)

class PairCountBase(object):
    """
    An abstract base class for pair counting algorithms, either for a
    simulation box or survey data.

    Users should use one of the subclasses of this class.

    Parameters
    ----------
    first : CatalogSource
        the first source of particles
    second : CatalogSource, optional
        the second source of particles to cross-correlate
    """
    def __init__(self, first, second, show_progress=True):

        self.first = first
        self.second = second
        self.comm = first.comm
        self.attrs = {}
        self.attrs['show_progress'] = show_progress

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

    def _log(self, N1, N2, pos1, pos2):
        """
        Internal function to log the distribution of particles to correlate
        across ranks.

        Parameters
        ----------
        pos1 : array_like
            the first set of particle positions this rank is correlating
        pos2 : array_like
            the second set of particle positions this rank is correlating
        """
        # global counts
        if self.comm.rank == 0:
            self.logger.info('correlating %d x %d objects in total' %(N1, N2))

        # sizes for all ranks
        sizes1 = self.comm.gather(len(pos1), root=0)
        sizes2 = self.comm.gather(len(pos2), root=0)

        # rank 0 logs
        if self.comm.rank == 0:
            args = (numpy.median(sizes1), numpy.median(sizes2))
            self.logger.info("correlating A x B = %d x %d objects (median) per rank" % args)

            global_min = numpy.min(sizes1)
            self.logger.info("min A load per rank = %d" % global_min)

            global_max = numpy.max(sizes1)
            self.logger.info("max A load per rank = %d" % global_max)

            args = (N1//self.comm.size, N2)
            self.logger.info("(even distribution would result in %d x %d)" % args)

    def _decompose(self):
        """
        An internal function to perform a domain decomposition on the two
        sources we wish to correlate, returning the position/weights for each
        object in the correlating pair.

        The particles are assumed to be in a cubical box, and no load
        balancing is required.

        The implementation follows:

        1. Decompose the first source such that the objects are spatially
           tight on a given rank.
        2. Decompose the second source, ensuring a given rank holds all
           particles within the desired maximum separation.

        Returns
        -------
        (pos1, w1), (pos2, w2) : array_like
            the (decomposed) set of positions and weights to correlate
        """
        from pmesh.domain import GridND
        from nbodykit.utils import split_size_3d

        # determine processor division for domain decomposition
        np = split_size_3d(self.comm.size)
        if self.comm.rank == 0:
            self.logger.info("using cpu grid decomposition: %s" %str(np))

        # get the (periodic-enforced) position for first
        pos1 = self.first['Position']
        if self.attrs['periodic']:
            pos1 %= self.attrs['BoxSize']
        pos1, w1 = self.first.compute(pos1, self.first[self.attrs['weight']])
        N1 = self.comm.allreduce(len(pos1))

        # get the (periodic-enforced) position for second
        if self.second is not None:
            pos2 = self.second['Position']
            if self.attrs['periodic']:
                pos2 %= self.attrs['BoxSize']
            pos2, w2 = self.second.compute(pos2, self.second[self.attrs['weight']])
            N2 = self.comm.allreduce(len(pos2))
        else:
            pos2 = pos1
            w2 = w1
            N2 = N1

        # domain decomposition
        grid = [
            numpy.linspace(0, self.attrs['BoxSize'][0], np[0] + 1, endpoint=True),
            numpy.linspace(0, self.attrs['BoxSize'][1], np[1] + 1, endpoint=True),
            numpy.linspace(0, self.attrs['BoxSize'][2], np[2] + 1, endpoint=True),
        ]
        domain = GridND(grid, comm=self.comm)

        # exchange first particles
        layout = domain.decompose(pos1, smoothing=0)
        pos1 = layout.exchange(pos1)
        w1 = layout.exchange(w1)

        # exchange second particles
        rmax = numpy.max(self.attrs['redges'])
        if rmax > self.attrs['BoxSize'].max() * 0.25:
            pos2 = numpy.concatenate(self.comm.allgather(pos2), axis=0)
            w2   = numpy.concatenate(self.comm.allgather(w2), axis=0)
        else:
            layout  = domain.decompose(pos2, smoothing=rmax)
            pos2 = layout.exchange(pos2)
            w2   = layout.exchange(w2)

        # log the sizes of the trees
        self._log(N1, N2, pos1, pos2)

        return (pos1, w1), (pos2, w2)

    def _run(self, func, kwargs, loads, callback=None):
        """
        Internal function that calls ``func`` in ``N`` iterations, optionally
        calling ``callback`` before each iteration.

        This allows the Corrfunc function ``func`` to be called in chunks,
        giving the user a progress report after each iteration.

        Parameters
        ----------
        func : callable
            the Corrfunc callable -- ``kwargs`` will be passed ot this function
        kwargs : dict
            the dictionary of arguments to pass to ``func``
        loads : list of int
            the list of loads for every rank; this corresponds to the number
            of particles in the in ``A`` if we are correlating ``A`` x ``B``
        callback : callable, optional
            a callable takings ``kwargs`` as its first argument and a slice
            object as its second argument; this will be called first during
            each iteration

        Returns
        -------
        pc : CorrfuncResult
            the total pair count result, summed over all iterations run
        """
        # the rank with the largest load
        largest_load = numpy.argmax(loads)

        # do the pair counting
        def run(chunk):
            if callback is not None:
                callback(kwargs, chunk)
            return self._run_corrfunc(func, **kwargs)

        # log the function start
        if self.comm.rank == 0:
            name = func.__module__ + '.' + func.__name__
            self.logger.info("calling function '%s'" % name)

        # number of iterations
        N = 10 if self.attrs['show_progress'] else 1

        # run in chunks
        pc = None
        chunks = numpy.array_split(range(loads[self.comm.rank]), N, axis=0)
        for i, chunk in enumerate(chunks):
            this_pc = run(chunk)
            if self.comm.rank == largest_load and self.attrs['show_progress']:
                self.logger.info("%d%% done" % (N*(i+1)))

            # sum up the results
            pc = this_pc if pc is None else pc + this_pc

        return pc

    def _run_corrfunc(self, func, **kws):
        """
        Run the wrapped Corrfunc function ``func``, passing in the keywords
        specified by ``kws``.

        This hides all output from the Corrfunc function (stdout, stderr,
        and C-level output), unless an exception occurs.
        """
        from nbodykit.utils import captured_output

        try:
            # record progress capture output for everything but root
            with captured_output(self.comm, root=None) as (out, err):
                result = func(**kws)

        except Exception as e:
            # get the values Corrfunc logged to stdout and stderr
            stdout = out.getvalue(); stderr = err.getvalue()

            # log all of the output in a new exception
            name = func.__module__ + '.' + func.__name__
            msg = "calling the function '%s' failed:\n" % name
            msg += "exception: %s\n" % str(e)
            msg += "stdout: %s\n" % stdout
            msg += "stderr: %s" % stderr
            raise RuntimeError(msg)

        return CorrfuncResult(result)

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
    first : CatalogSource
        the first source of particles, providing the 'Position' column
    redges : array_like
        the radius bin edges; length of nbins+1
    BoxSize : float, 3-vector, optional
        the size of the box; if 'BoxSize' is not provided in the source
        'attrs', it must be provided here
    second : CatalogSource, optional
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
    show_progress : bool, optional
        if ``True``, perform the pair counting calculation in 10 iterations,
        logging the progress after each iteration; this is useful for
        understanding the scaling of the code
    **config : key/value pairs
        additional keywords to pass to the :mod:`Corrfunc` function
    """
    logger = logging.getLogger('SimulationBoxPairCount')

    def __init__(self, mode, first, redges, BoxSize=None, periodic=True,
                    second=None, Nmu=5, los='z', weight='Weight',
                    show_progress=True, **config):

        # check input 'mode'
        assert mode in ['1d', '2d'], "PairCount mode must be '1d' or '2d'"

        # check rmin
        if numpy.min(redges) <= 0.:
            raise ValueError(("the lower edge of the 1st separation bin must "
                              "greater than zero (no self-pairs)"))

        # check input 'los'
        if isinstance(los, string_types):
            assert los in 'xyz', "``los`` should be one of 'x', 'y', 'z'"
            los = 'xyz'.index(los)
        elif isinstance(los, int):
            if los < 0: los += 3
            assert los in [0,1,2], "``los`` should be one of 0, 1, 2"
        else:
            raise ValueError("``los`` should be either ['x', 'y', 'z'] or [0,1,2]")

        # verify the input sources
        BoxSize = verify_input_sources(first, second, BoxSize, ['Position', weight])

        # init the base class
        PairCountBase.__init__(self, first, second, show_progress)

        # save the meta-data
        self.attrs['mode'] = mode
        self.attrs['redges'] = redges
        self.attrs['Nmu'] = Nmu
        self.attrs['BoxSize'] = BoxSize
        self.attrs['periodic'] = periodic
        self.attrs['weight'] = weight
        self.attrs['config'] = config
        self.attrs['los'] = los

        # test Rmax for PBC
        if periodic and numpy.amax(redges) > 0.5*self.attrs['BoxSize'].min():
            raise ValueError(("periodic pair counts cannot be computed for "
                              "Rmax > 0.5 * BoxSize"))

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
        try:
            from Corrfunc.theory import DD, DDsmu
        except ImportError:
            raise MissingCorrfuncError()

        # some setup
        redges = self.attrs['redges']
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
                raise ValueError(("``Corrfunc`` does not currently support  "
                                  "periodic wrapping with non-cubic boxes"))

        # domain decompose the data
        (pos1, w1), (pos2, w2) = self._decompose()

        # total sizes for first and second
        N1 = self.comm.allreduce(len(pos1))
        N2 = self.comm.allreduce(len(pos2))

        # initialize the keywords to pass to Corrfunc
        kws = {}
        kws['autocorr'] = 0
        kws['nthreads'] = 1
        kws['binfile'] = redges
        kws['X2'] = pos2[:,perp_axes[0]]
        kws['Y2'] = pos2[:,perp_axes[1]]
        kws['Z2'] = pos2[:,los]
        kws['weights2'] = w2.astype(pos2.dtype)
        kws['weight_type'] = 'pair_product'
        kws['periodic'] = self.attrs['periodic']
        if L is not None: kws['boxsize'] = L

        # 1D calculation
        if self.attrs['mode'] == '1d':
            kws['output_ravg'] = True
            func = DD
            rcol = 'ravg'
        # 2D calculation
        else:
            kws['output_savg'] = True
            kws['mu_max'] = 1.0
            kws['nmu_bins'] = self.attrs['Nmu']
            func = DDsmu
            rcol = 'savg'

        # update with user config
        kws.update(self.attrs['config'])

        def callback(kws, chunk):
            kws['X1'] = pos1[chunk][:,perp_axes[0]]
            kws['Y1'] = pos1[chunk][:,perp_axes[1]]
            kws['Z1'] = pos1[chunk][:,los] # LOS defined with respect to this axis
            kws['weights1'] = w1[chunk].astype(pos1.dtype)

        # run
        sizes = self.comm.allgather(len(pos1))
        pc = self._run(func, kws, sizes, callback=callback)

        # make 2D results shape of (Nr, Nmu)
        if self.attrs['mode'] == '2d':
            pc = pc.reshape((-1, self.attrs['Nmu']))

        # sum results over all ranks
        pc = self.comm.allreduce(pc)

        # the output data
        dtype = numpy.dtype([('r', 'f8'), ('xi', 'f8'), ('npairs', 'u8'), ('weightavg', 'f8')])
        data = numpy.empty(pc.shape, dtype=dtype)

        # copy over main results
        data['r'] = pc[rcol]
        data['npairs'] = pc['npairs']
        data['weightavg'] = pc['weightavg']

        # compute the random pairs from the fractional volume
        RR = 1.*N1*N2 / boxsize.prod()
        if self.attrs['mode'] == '2d':
            dr3 = numpy.diff(redges**3)
            muedges = numpy.linspace(0, 1, self.attrs['Nmu']+1)
            dmu = numpy.diff(muedges)
            RR *= 2. / 3. * numpy.pi * dr3[:,None] * dmu[None,:]
        else:
            RR *= 4. / 3. * numpy.pi * numpy.diff(redges**3)

        # add in the correlation function value
        data['xi'] = (1. * data['npairs'] / RR) - 1.0

        # make the BinnedStatistic
        if self.attrs['mode'] == '1d':
            self.result = BinnedStatistic(['r'], [redges], data, fields_to_sum=['npairs'])
        else:
            self.result = BinnedStatistic(['r','mu'], [redges,muedges], data, fields_to_sum=['npairs'])


def verify_input_sources(first, second, BoxSize, required_columns, inspect_boxsize=True):
    """
    Verify that the input source objects have all of the required columns
    and appropriate box size attributes.
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
