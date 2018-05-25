from numpy.random import RandomState
import numpy
from nbodykit.utils import FrontPadArray

class MPIRandomState:
    """ A Random number generator that is invariant against number of ranks,
        when the total size of random number requested is kept the same.

        The algorithm here assumes the random number generator from numpy
        produces uncorrelated results when the seeds are sampled from a single
        RNG.

        The sampler methods are collective calls; multiple calls will return
        uncorrerlated results.

        The result is only invariant under diif comm.size when allreduce(size)
        and chunksize are kept invariant.

    """
    def __init__(self, comm, seed, size, chunksize=100000):
        self.comm = comm
        self.seed = seed
        self.chunksize = chunksize

        self.size = size
        self.csize = numpy.sum(comm.allgather(size), dtype='intp')

        self._start = numpy.sum(comm.allgather(size)[:comm.rank], dtype='intp')
        self._end = self._start + self.size

        self._first_ichunk = self._start // chunksize

        self._skip = self._start - self._first_ichunk * chunksize

        nchunks = (comm.allreduce(numpy.array(size, dtype='intp')) + chunksize - 1) // chunksize
        self.nchunks = nchunks

        self._serial_rng = RandomState(seed)

    def _prepare_args_and_result(self, args, itemshape, dtype):
        """ pad every item in args with values from previous ranks,
            and create an array for holding the result with the same length.

            Returns
            -------
            padded_r, padded_args

        """

        r = numpy.zeros((self.size,) + tuple(itemshape), dtype=dtype)

        r_and_args = (r,) + tuple(args)
        r_and_args_b = numpy.broadcast_arrays(*r_and_args)

        padded = []

        # we don't need to pad scalars,
        # loop over broadcasted and non broadcast version to figure this out)
        for a, a_b in zip(r_and_args, r_and_args_b):
            if numpy.isscalar(a):
                # use the scalar, no need to pad.
                padded.append(a)
            else:
                # not a scalar, pad
                padded.append(FrontPadArray(a_b, self._skip, self.comm))

        return padded[0], padded[1:]

    def poisson(self, lam, itemshape=(), dtype='f8'):
        """ Produce `self.size` poissons, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            lam, = args
            return rng.poisson(lam=lam, size=size)
        return self._call_rngmethod(sampler, (lam,), itemshape, dtype)

    def choice(self, choices, itemshape=(), replace=True, p=None):
        """ Produce `self.size` choices, each of shape itemshape. This is a collective MPI call. """
        dtype = numpy.array(choices).dtype
        def sampler(rng, args, size):
            return rng.choice(choices, size=size, replace=replace, p=p)

        return self._call_rngmethod(sampler, (), itemshape, dtype)

    def normal(self, loc=0, scale=1, itemshape=(), dtype='f8'):
        """ Produce `self.size` normals, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            loc, scale = args
            return rng.normal(loc=loc, scale=scale, size=size)
        return self._call_rngmethod(sampler, (loc, scale), itemshape, dtype)

    def uniform(self, low=0., high=1.0, itemshape=(), dtype='f8'):
        """ Produce `self.size` uniforms, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            low, high = args
            return rng.uniform(low=low, high=high,size=size)
        return self._call_rngmethod(sampler, (low, high), itemshape, dtype)

    def _call_rngmethod(self, sampler, args, itemshape, dtype='f8'):
        """
            Loop over the seed table, and call sampler(rng, args, size)
            on each rng, with matched input args and size.

            the args are padded in the front such that the rng is invariant
            no matter how self.size is distributed.

            truncate the return value at the front to match the requested `self.size`.
        """

        seeds = self._serial_rng.randint(0, high=0xffffffff, size=self.nchunks)

        padded_r, running_args = self._prepare_args_and_result(args, itemshape, dtype)

        running_r = padded_r
        ichunk = self._first_ichunk

        while len(running_r) > 0:
            # at most get a full chunk, or the remaining items
            nreq = min(len(running_r), self.chunksize)

            seed = seeds[ichunk]
            rng = RandomState(seed)
            args = tuple([a if numpy.isscalar(a) else a[:nreq] for a in running_args])

            # generate nreq random items from the sampler 
            chunk = sampler(rng, args=args,
                size=(nreq,) + tuple(itemshape))

            running_r[:nreq] = chunk

            # update running arrays, since we have finished nreq items
            running_r = running_r[nreq:]
            running_args = tuple([a if numpy.isscalar(a) else a[nreq:] for a in running_args])

            ichunk = ichunk + 1

        return padded_r[self._skip:]

