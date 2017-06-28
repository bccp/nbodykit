import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.dataset import DataSet

def verify_input_sources(first, second, BoxSize, required_columns):
    """
    Verify the input source objects have
    """
    if second is None: second = first

    # check for comm mismatch
    assert second.comm is first.comm, "communicator mismatch between input sources"

    _BoxSize = first.attrs['BoxSize'].copy()
    if BoxSize is not None:
        _BoxSize[:] = BoxSize

    # check box sizes
    if not numpy.array_equal(first.attrs['BoxSize'], second.attrs['BoxSize']):
        raise ValueError("BoxSize mismatch between pair count cross-correlation sources")
    if not numpy.array_equal(first.attrs['BoxSize'], _BoxSize):
        raise ValueError("BoxSize mismatch between sources and the pair count algorithm")

    # check required columns
    for source in [first, second]:
        for col in required_columns:
            if col not in source:
                raise ValueError("the column '%s' is missing from input source; cannot do pair count" %col)

    return _BoxSize

class SimulationBoxPairCount(object):
    """
    Count (weighted) pairs of objects in a simulation box using the :mod:`Corrfunc` package

    This uses the :func:`Corrfunc.theory.DD` function to count pairs
    """
    logger = logging.getLogger('SimulationBoxPairCount')

    def __init__(self, source1, edges, BoxSize=None, periodic=True, weight='Weight', source2=None, **config):
        """
        Parameters
        ----------
        source1 : CatalogSource
            the first source of particles, providing the 'Position' column
        edges : array_like
            the radius bin edges; length of nbins+1
        BoxSize : float, 3-vector; optional
            the size of the box; if 'BoxSize' is not provided in the source
            'attrs', it must be provided here
        periodic : bool; optional
            whether to use periodic boundary conditions
        weight : str; optional
            the name of the column in the source specifying the particle weights
        source2 : CatalogSource; optional
            the second source of particles to cross-correlate
        **config : key/value pairs
            additional keywords to pass to the :func:`Corrfunc.theory.DD`
            function
        """
        # verify the input sources
        BoxSize = verify_input_sources(source1, source2, BoxSize, ['Position', weight])

        self.source1 = source1
        self.source2 = source2
        self.comm = source1.comm

        # save the meta-data
        self.attrs = {}
        self.attrs['BoxSize'] = BoxSize
        self.attrs['edges'] = edges
        self.attrs['periodic'] = periodic
        self.attrs['weight'] = weight
        self.attrs['config'] = config

        # test Rmax for PBC
        if periodic and numpy.amax(edges) > 0.5*self.attrs['BoxSize'].min():
            raise ValueError("periodic pair counts cannot be computed for Rmax > 0.5 * BoxSize")

        # run the algorithm
        self.run()

    def run(self):
        """
        Calculate the 3D pair-counts

        Attributes
        ----------
        result : :class:`~nbodykit.dataset.DataSet`
            a DataSet object holding the pair count and correlation function results
        """
        try:
            import Corrfunc
        except:
            raise ImportError("please download and install ``Corrfunc`` from ``https://github.com/nickhand/Corrfunc``")
        from pmesh.domain import GridND

        # some setup
        redges = self.attrs['edges']
        comm   = self.comm
        nbins  = len(redges)-1
        boxsize = self.attrs['BoxSize']
        L = None

        # cubic box required by Corrunc
        if self.attrs['periodic'] and numpy.all(boxsize == boxsize[0]):
            L = boxsize[0]
        elif self.attrs['periodic']:
            raise ValueError("``Corrfunc`` does not currently support periodic wrapping with non-cubic boxes")

        # determine processor division for domain decomposition
        for Nx in range(int(comm.size**0.3333) + 1, 0, -1):
            if comm.size % Nx == 0: break
        else:
            Nx = 1
        for Ny in range(int(comm.size**0.5) + 1, 0, -1):
            if (comm.size // Nx) % Ny == 0: break
        else:
            Ny = 1
        Nz = comm.size // Nx // Ny
        Nproc = [Nx, Ny, Nz]
        if self.comm.rank == 0:
            self.logger.info("using cpu grid decomposition: %s" %str(Nproc))

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
        grid = [numpy.linspace(posmin[i], posmax[i], Nproc[i]+1, endpoint=True) for i in range(3)]
        domain = GridND(grid, comm=comm)

        layout = domain.decompose(pos1, smoothing=0)
        pos1   = layout.exchange(pos1)
        w1     = layout.exchange(w1)

        # get the position/weight of the secondaries
        rmax = numpy.max(self.attrs['edges'])
        if rmax > self.attrs['BoxSize'].max() * 0.25:
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
        kws['X2'] = pos2[:,0]
        kws['Y2'] = pos2[:,1]
        kws['Z2'] = pos2[:,2]
        kws['weights2'] = w2.astype(pos2.dtype)
        kws['weight_type'] = 'pair_product'
        kws['output_ravg'] = True
        if L is not None: kws['boxsize'] = L
        kws.update(self.attrs['config'])

        pc = Corrfunc.theory.DD(0, 1, redges, pos1[:,0], pos1[:,1], pos1[:,2], **kws)
        self.logger.info('...rank %d done correlating' %(comm.rank))

        # make a new structured array
        dtype = numpy.dtype([('r', 'f8'), ('xi', 'f8'), ('npairs', 'f8'), ('weightavg', 'f8')])
        data = numpy.empty(len(pc), dtype=dtype)

        # do the sum across ranks
        data['r'][:] = comm.allreduce(pc['npairs']*pc['ravg'])
        data['weightavg'][:] = comm.allreduce(pc['npairs']*pc['weightavg'])
        data['npairs'][:] = comm.allreduce(pc['npairs'])
        idx = data['npairs'] > 0.
        data['r'][idx] /= data['npairs'][idx]
        data['weightavg'][idx] /= data['npairs'][idx]

        # compute the random pairs from the fractional volume
        RR = 1.*N1*N2 / boxsize.prod()
        RR *= 4. / 3. * numpy.pi * numpy.diff(redges**3)
        data['xi'] = (1. * data['npairs'] / RR) - 1.0

        # make the DataSet
        self.result = DataSet(['r'], [redges], data, fields_to_sum=['npairs'])

    def __getstate__(self):
        return {'result':self.result.data, 'attrs':self.attrs}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.result = DataSet(['r'], [self.attrs['edges']], self.result, fields_to_sum=['npairs'])

    def save(self, output):
        """
        Save result as a JSON file
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
