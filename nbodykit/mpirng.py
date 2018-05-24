from numpy.random import RandomState
import numpy
from nbodykit.utils import FrontPadArray

class MPIRandomState:
    """ A Random number generator that is invariant against number of ranks,
        when the total size of random number requested is kept the same.

        The algorithm here assumes the random number generator from numpy
        produces uncorrelated results when the seeds are sampled from a single
        RNG.

        The sampler methods are collective calls.

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

        rng = RandomState(seed)
        self._seeds = rng.randint(0, high=0xffffffff, size=nchunks)

    def prepare_args_and_result(self, args, itemshape, dtype):
        r = numpy.zeros((self.size,) + tuple(itemshape), dtype=dtype)

        r_and_args = numpy.broadcast_arrays(r, *args)

        padded = [FrontPadArray(a, self._skip, self.comm) for a in r_and_args]
        return padded[0], padded[1:]

    def poisson(self, lam, itemshape=(), dtype='f8'):
        """ Produce `self.size` poissons. This is a collective MPI call. """
        def func(rng, args, size):
            lam, = args
            return rng.poisson(lam=lam, size=size)
        return self._call_rngmethod(func, (lam,), itemshape, dtype)

    def uniform(self, low=0., high=1.0, itemshape=(), dtype='f8'):
        """ Produce `self.size` uniforms. This is a collective MPI call. """
        def func(rng, args, size):
            low, high = args
            return rng.uniform(low=low, high=high,size=size)
        return self._call_rngmethod(func, (low, high), itemshape, dtype)

    def _call_rngmethod(self, func, args, itemshape, dtype='f8'):
        """
            Loop over the seed table, and call func(rng, args, size)
            on each rng, with matched input args and size.

            the args are padded in the front such that the rng is invariant
            no matter how self.size is distributed.

            truncate the return value at the front to match the requested `self.size`.
        """

        padded_r, running_args = self.prepare_args_and_result(args, itemshape, dtype)

        running_r = padded_r
        ichunk = self._first_ichunk

        while len(running_r) > 0:
            nreq = min(len(running_r), self.chunksize)
            seed = self._seeds[ichunk]
            rng = RandomState(seed)
            args = tuple([a[:nreq] for a in running_args])

            firstchunk = func(rng, args=args,
                size=(nreq,) + tuple(itemshape))

            running_r[:nreq] = firstchunk
            running_r = running_r[nreq:]
            running_args = tuple([a[nreq:] for a in running_args])

            ichunk = ichunk + 1

        return padded_r[self._skip:]

