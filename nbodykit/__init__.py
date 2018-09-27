from .version import __version__

from mpi4py import MPI

import dask

import warnings


try:
    # prevents too many threads exception when using MPI and dask
    # by disabling threading in dask.
    dask.config.set(scheduler='synchronous') 
except:
    # deprecated since 0.18.1
    dask.set_options(get=dask.get)

_global_options = {}
_global_options['global_cache_size'] = 1e8 # 100 MB
_global_options['dask_chunk_size'] = 100000
_global_options['paint_chunk_size'] = 1024 * 1024 * 4

from contextlib import contextmanager
import logging

class CurrentMPIComm(object):
    """
    A class to faciliate getting and setting the current MPI communicator.
    """
    _stack = [MPI.COMM_WORLD]
    logger = logging.getLogger("CurrentMPIComm")

    @staticmethod
    def enable(func):
        """
        Decorator to attach the current MPI communicator to the input
        keyword arguments of ``func``, via the ``comm`` keyword.
        """
        import functools
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            kwargs.setdefault('comm', None)
            if kwargs['comm'] is None:
                kwargs['comm'] = CurrentMPIComm.get()
            return func(*args, **kwargs)
        return wrapped

    @classmethod
    @contextmanager
    def enter(cls, comm):
        """
        Enters a context where the current default MPI communicator is modified to the
        argument `comm`. After leaving the context manager the communicator is restored.

        Example:

        .. code ::

            with CurrentMPIComm.enter(comm):
                cat = UniformCatalog(...)

        is identical to 

        .. code ::

            cat = UniformCatalog(..., comm=comm)

        """
        cls.push(comm)

        yield

        cls.pop()

    @classmethod
    def push(cls, comm):
        """ Switch to a new current default MPI communicator """
        cls._stack.append(comm)
        if comm.rank == 0:
            cls.logger.info("Entering a current communicator of size %d" % comm.size)
        cls._stack[-1].barrier()
    @classmethod
    def pop(cls):
        """ Restore to the previous current default MPI communicator """
        comm = cls._stack[-1]
        if comm.rank == 0:
            cls.logger.info("Leaving current communicator of size %d" % comm.size)
        cls._stack[-1].barrier()
        cls._stack.pop()
        comm = cls._stack[-1]
        if comm.rank == 0:
            cls.logger.info("Restored current communicator to size %d" % comm.size)

    @classmethod
    def get(cls):
        """
        Get the default current MPI communicator. The initial value is ``MPI.COMM_WORLD``.
        """
        return cls._stack[-1]

    @classmethod
    def set(cls, comm):
        """
        Set the current MPI communicator to the input value.
        """

        warnings.warn("CurrentMPIComm.set is deprecated. Use `with CurrentMPIComm.enter(comm):` instead")
        cls._stack[-1].barrier()
        cls._stack[-1] = comm
        cls._stack[-1].barrier()

class GlobalCache(object):
    """
    A class to faciliate calculation using a global cache via
    :class:`dask.cache.Cache`.
    """
    _instance = None

    @classmethod
    def get(cls):
        """
        Return the global cache object. The default size is controlled
        by the ``global_cache_size`` global option; see :class:`set_options`.

        Returns
        -------
        cache : :class:`dask.cache.Cache`
            the cache object, as provided by dask
        """
        # if not created, use default cache size
        if not cls._instance:
            from dask.cache import Cache
            cls._instance = Cache(_global_options['global_cache_size'])

        return cls._instance

    @classmethod
    def resize(cls, size):
        """
        Re-size the global cache to the specified size in bytes.

        Parameters
        ----------
        size : int, optional
            the desired size of the returned cache in bytes; if not provided,
            the ``global_cache_size`` global option is used
        """
        # get the cachey Cache
        # NOTE: cachey cache stored as the cache attribute of Dask cache
        cache = cls.get().cache

        # set the new size
        cache.available_bytes = size

        # shrink the cache if we need to
        # NOTE: only removes objects if we need to
        cache.shrink()

class set_options(object):
    """
    Set global configuration options.

    Parameters
    ----------
    dask_chunk_size : int
        the number of elements for the default chunk size for dask arrays;
        chunks should usually hold between 10 MB and 100 MB
    global_cache_size : float
        the size of the internal dask cache in bytes; default is 1e9
    paint_chunk_size : int
        the number of objects to paint at the same time. This is independent
        from dask chunksize.
    """
    def __init__(self, **kwargs):
        self.old = _global_options.copy()
        for key in sorted(kwargs):
            if key not in _global_options:
                raise KeyError("Option `%s` is not supported" % key)
        _global_options.update(kwargs)

        # resize the global Cache!
        self.updated_cache_size = False
        if 'global_cache_size' in kwargs:
            GlobalCache.resize(_global_options['global_cache_size'])
            self.updated_cache_size = True

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        _global_options.clear()
        _global_options.update(self.old)

        # resize Cache to original size
        if self.updated_cache_size:
            GlobalCache.resize(_global_options['global_cache_size'])


_logging_handler = None
def setup_logging(log_level="info"):
    """
    Turn on logging, with the specified level.

    Parameters
    ----------
    log_level : 'info', 'debug', 'warning'
        the logging level to set; logging below this level is ignored
    """

    # This gives:
    #
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Nproc = [2, 1, 1]
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Rmax = 120
    import logging

    levels = {
            "info" : logging.INFO,
            "debug" : logging.DEBUG,
            "warning" : logging.WARNING,
            }

    import time
    logger = logging.getLogger();
    t0 = time.time()

    rank = MPI.COMM_WORLD.rank

    class Formatter(logging.Formatter):
        def format(self, record):
            s1 = ('[ %09.2f ] % 3d: ' % (time.time() - t0, rank))
            return s1 + logging.Formatter.format(self, record)

    fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M ')

    global _logging_handler
    if _logging_handler is None:
        _logging_handler = logging.StreamHandler()
        logger.addHandler(_logging_handler)

    _logging_handler.setFormatter(fmt)
    logger.setLevel(levels[log_level])
