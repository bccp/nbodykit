from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic
import numpy

class PairCountBase(object):
    """
    An abstract base class for pair counting algorithms.

    Users should use one of the subclasses of this class.
    """
    def __init__(self, mode, edges, first, second, Nmu, pimax, weight, show_progress=False):

        # check input 'mode'
        valid_modes = ['1d', '2d', 'projected', 'angular']
        if mode not in valid_modes:
            raise ValueError("allowed 'mode' values are: %s" % valid_modes)

        # check min edge
        if numpy.min(edges) <= 0.:
            raise ValueError(("the lower edge of the 1st separation bin must "
                              "greater than zero (no self-pairs)"))

        # check mode requirements
        if mode == '2d' and Nmu is None:
            raise ValueError("'Nmu' keyword is required when 'mode' is '2d'")
        if Nmu is not None and mode != '2d':
            raise ValueError("mode should be '2d' if 'Nmu' is specified")
        if mode == 'projected' and pimax is None:
            raise ValueError("'pimax' keyword is required when 'mode' is 'projected'")
        if pimax is not None and mode != 'projected':
            raise ValueError("mode should be 'projected' if 'projected' is specified")
        if mode == 'projected' and pimax < 1.0:
            raise ValueError("'pimax' must be at least 1.0 when 'mode' is 'projected'")

        self.first = first
        self.second = second
        self.comm = first.comm

        # store the meta-data
        self.attrs = {}
        self.attrs['mode'] = mode
        self.attrs['edges'] = edges
        self.attrs['Nmu'] = Nmu
        self.attrs['pimax'] = pimax
        self.attrs['show_progress'] = show_progress

        # store the total size of the sources
        self.attrs['N1'] = first.csize
        self.attrs['N2'] = second.csize if second is not None else None

        if second is None or second is first:
            wpairs1, wpairs2 = self.comm.allreduce(first.compute(first[weight].sum())), self.comm.allreduce(first.compute((first[weight]**2).sum()))
            # for auto excluding self pairs to avoid a biased estimator. The factor 0.5 is by convention.
            # In the end it will cancel out in two point function estimators.
            self.attrs['total_wnpairs'] = 0.5*(wpairs1**2-wpairs2)
            self.attrs['is_cross'] = False
        else:
            wpairs1, wpairs2 = self.comm.allreduce(first.compute(first[weight].sum())), self.comm.allreduce(second.compute(second[weight].sum()))
            self.attrs['total_wnpairs'] = 0.5*wpairs1*wpairs2
            self.attrs['is_cross'] = True

    def __getstate__(self):
        return {'pairs':self.pairs.data, 'attrs':self.attrs}

    def __setstate__(self, state):
        self.__dict__.update(state)
        edges = self.attrs['edges']

        # reconstruct the result based on mode
        kws = {'fields_to_sum' : ['npairs', 'wnpairs']}
        if self.attrs['mode'] == '1d':
            dims, edges = ['r'], [edges]

        elif self.attrs['mode'] == '2d':
            muedges = numpy.linspace(0, 1., self.attrs['Nmu']+1)
            dims, edges = ['r', 'mu'], [edges, muedges]

        elif self.attrs['mode'] == 'projected':
            piedges = numpy.linspace(0, self.attrs['pimax'], self.attrs['pimax']+1)
            dims, edges = ['rp', 'pi'], [edges, piedges]

        elif self.attrs['mode'] == 'angular':
            dims, edges = ['theta'], [edges]

        else:
            valid = ['1d','2d','angular','projected']
            args = (self.attrs['mode'], valid)
            raise ValueError("mode = '%s' should be one of %s" % args)

        # save the result as a BinnedStatistic
        self.pairs = BinnedStatistic(dims, edges, self.pairs, **kws)

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
            raise ValueError("BoxSize must be supplied in the source ``attrs`` or via the ``BoxSize`` keyword")

        BoxSize2 = second.attrs.get('BoxSize', None)
        if BoxSize1 is not None and BoxSize2 is not None:
            if not numpy.array_equal(first.attrs['BoxSize'], BoxSize2):
                raise ValueError("BoxSize mismatch between pair count cross-correlation sources")
            if not numpy.array_equal(first.attrs['BoxSize'], _BoxSize):
                raise ValueError("BoxSize mismatch between sources and the pair count algorithm")

        return _BoxSize
