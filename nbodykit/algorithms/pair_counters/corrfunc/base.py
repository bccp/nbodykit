import numpy
import logging
from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic

class MissingCorrfuncError(Exception):
    def __init__(self):
        msg = "use either ``conda install -c bccp corrfunc`` "
        msg += "or ``pip install git+git://github.com/nickhand/Corrfunc``"
        self.args = (msg,)

class CorrfuncResult(object):
    """
    A class used internally to hold the array-like result of a
    pair counting algorithm from :mod:`Corrfunc`.

    This class is useful for reducing pair count results in parallel
    while accounting for columns that are pair-weighted.

    Parameters
    ----------
    data : numpy.ndarray
        the numpy structured array result from :mod:`Corrfunc`
    """
    valid = ['weightavg', 'npairs', 'savg', 'ravg', 'thetaavg', 'rpavg']

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

def tonativeendian(arr):
    arr = arr.astype(arr.dtype.newbyteorder('='))
    return arr

class MPICorrfuncCallable(object):
    """
    A base class to represent an MPI-enabled :mod:`Corrfunc` callable.

    This class adds the following functionality to the Corrfunc code:

    - If ``show_progress`` is ``True``, execute the function in chunks,
      logging to screen the progress along the way. This is useful for
      potentially long running pair counting jobs.
    - When calling the function, capture stdout/stderr and C-level output
      and if an error occurs, raise an exception with all generated output for
      the user.

    The relevant subclasses of this class are in :mod:`mocks` and :mod:`theory`.
    """
    binning_dims = None
    logger = logging.getLogger("MPICorrfuncCallable")

    def __init__(self, callable, comm, show_progress=True):

        self.callable = callable
        self.comm = comm
        self.show_progress = show_progress

    def __call__(self, loads, kwargs, callback=None):
        """
        Calls :attr:`callable` in iterations, optionally
        calling ``callback`` before each iteration.

        This allows the :mod:`Corrfunc` function :attr:`callable` to be called
        in chunks, giving the user a progress report after each iteration.

        Parameters
        ----------
        loads : list of int
            the list of loads for every rank; this corresponds to the number
            of particles in the in ``A`` if we are correlating ``A`` x ``B``
        kwargs : dict
            the dictionary of arguments to pass to ``func``
        callback : callable, optional
            a callable takings ``kwargs`` as its first argument and a slice
            object as its second argument; this will be called first during
            each iteration

        Returns
        -------
        result : BinnedStatistic
            the total binned pair counting result
        """
        # the rank with the largest load
        largest_load = numpy.argmax(loads)

        # do the pair counting
        def run(chunk):
            if callback is not None:
                callback(kwargs, chunk)
            return self._run(self.callable, kwargs)

        # log the function start
        if self.comm.rank == 0:
            name = self.callable.__module__ + '.' + self.callable.__name__
            self.logger.info("calling function '%s'" % name)

        # number of iterations
        N = 10 if self.show_progress else 1

        # run in chunks
        pc = None
        chunks = numpy.array_split(numpy.arange(loads[self.comm.rank],dtype='intp'), N, axis=0)
        for i, chunk in enumerate(chunks):
            this_pc = run(chunk)
            if self.comm.rank == largest_load and self.show_progress:
                self.logger.info("%d%% done" % (N*(i+1)))

            # sum up the results
            pc = this_pc if pc is None else pc + this_pc

        # convert flattened 1D results to 2D array
        if len(self.edges) > 1:
            pc = pc.reshape((-1, len(self.edges[1])-1))

        # reduce the result across all ranks
        pc = self.comm.allreduce(pc)

        # the dimension names (use "r" instead of "s")
        dims = list(self.binning_dims) # make a copy here
        if 's' in dims: dims[dims.index('s')] = 'r'

        # make a new structured array
        dtype = numpy.dtype([(dims[0], 'f8'),
                    ('npairs', 'u8'),
                    ('wnpairs', 'f8')])
        data = numpy.zeros(pc.shape, dtype=dtype)

        # copy over main results
        data[dims[0]] = pc[self.binning_dims[0]+'avg']
        data['npairs'] = pc['npairs']
        data['wnpairs'] = pc['weightavg'] * pc['npairs']
        # patch up when there is no pair, the weighted value shall be zero as well.
        data['wnpairs'][pc['npairs'] == 0] = 0

        # return the BinnedStatistic
        return BinnedStatistic(dims, self.edges, data, fields_to_sum=['npairs', 'wnpairs'])

    def _run(self, func, kws):
        """
        Internal function to run the wrapped :mod:`Corrfunc` function
        :attr:`func`, passing in the keywords specified by ``kws``.

        .. note::
            This hides all output from the Corrfunc function (stdout, stderr,
            and C-level output), unless an exception occurs.
        """
        from nbodykit.utils import captured_output

        kws = kws.copy()

        # cast the array items to the native endianness
        # because corrfunc doesn't understand otherwise.
        # https://github.com/bccp/nbodykit/issues/467

        for key, value in kws.items():
            if isinstance(value, numpy.ndarray):
                kws[key] = tonativeendian(value)

        try:
            # record progress capture output for everything but root
            with captured_output(self.comm, root=None) as (out, err):
                result = func(**kws)

        except Exception as e:
            # get the values Corrfunc logged to stdout and stderr
            stdout = out.getvalue(); stderr = err.getvalue()

            # log all of the output in a new exception
            name = func.__module__ + '.' + func.__name__
            msg = "calling the function '%s' failed, " % name
            msg += "likely due to issues with input data/parameters. "
            msg += "Open at issue at https://github.com/bccp/nbodykit/issues for further help.\n"
            msg += "exception: %s\n" % str(e)
            msg += "stdout: %s\n" % stdout
            msg += "stderr: %s" % stderr
            raise RuntimeError(msg)

        return CorrfuncResult(result)
