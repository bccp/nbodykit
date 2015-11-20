import logging
import sys

class Filter(logging.Filter):
    def filter(self, record):
        on = record.on
        comm = record.comm
        if on is None or on == comm.rank:
            return True
        return False

class MPILoggerAdapter(logging.LoggerAdapter):
    """ A logger adapter that allows a single rank to log. 
        
        The log method takes additional arguments:

        on : None or int
            rank to record the message
        comm : communicator 
            (defaults to MPI.COMM_WORLD)

    """
    def __init__(self, obj):
        logging.LoggerAdapter.__init__(self, obj, {})
        obj.addFilter(Filter())

    def process(self, msg, kwargs):
        if 'mpi4py' in sys.modules:
            from mpi4py import MPI 
            on   = kwargs.pop('on', None)
            comm = kwargs.pop('comm', MPI.COMM_WORLD)
            hostname = MPI.Get_processor_name()
            format='rank %(rank)d on %(hostname)-12s '
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            d = kwargs['extra']
            d['on'] = on
            d['comm'] = comm
            d['rank'] = comm.rank
            d['hostname'] = hostname.split('.')[0]
            return ((format % d) + msg, kwargs)
        else:
            return (msg, kwargs)
    def setLevel(self, level):
        self.logger.setLevel(level)

