__version__ = "1.0.0.dev0"

from mpi4py import MPI

class CurrentMPIComm(object):
    """
    The current MPI communicator
    """
    _instance = None

    @staticmethod
    def enable(func):
        import functools
        functools.wraps(func)
        def wrapped(*args, **kwargs):
            kwargs.setdefault('comm', None)
            if kwargs['comm'] is None:
                kwargs['comm'] = CurrentMPIComm.get()
            return func(*args, **kwargs)
        return wrapped

    @classmethod
    def get(cls):
        """
        Get the communicator, return ``MPI.COMM_WORLD``
        if the comm has not be explicitly set yet
        """
        # initialize MPI and set the comm if we need to
        if not cls._instance:
            comm = MPI.COMM_WORLD
            cls._instance = comm

        return cls._instance

    @classmethod
    def set(cls, comm):
        """
        Set the communicator to the input value
        """
        cls._instance = comm

def setup_logging(log_level="info"):
    """
    Set the basic configuration of all loggers
    """

    # This gives:
    #
    # [ 000000.43 ]   0:waterfall 06-28 14:49  measurestats    INFO     Nproc = [2, 1, 1]
    # [ 000000.43 ]   0:waterfall 06-28 14:49  measurestats    INFO     Rmax = 120
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
    name = MPI.Get_processor_name().split('.')[0]

    class Formatter(logging.Formatter):
        def format(self, record):
            s1 = ('[ %09.2f ] % 3d:%s ' % (time.time() - t0, rank, name))
            return s1 + logging.Formatter.format(self, record)

    fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M ')

    hdlr = logging.StreamHandler()
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    logger.setLevel(levels[log_level])


from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
