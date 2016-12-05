from mpi4py import MPI

import numpy
from astropy import cosmology
import logging
   
from nbodykit.batch import TaskManager 
from nbodykit import io
from nbodykit import source as Source
from nbodykit import algorithms

def setup_logging(log_level=logging.INFO):
    """
    Set the basic configuration of all loggers
    """

    # This gives:
    #
    # [ 000000.43 ]   0:waterfall 06-28 14:49  measurestats    INFO     Nproc = [2, 1, 1]
    # [ 000000.43 ]   0:waterfall 06-28 14:49  measurestats    INFO     Rmax = 120
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
    logger.setLevel(log_level)