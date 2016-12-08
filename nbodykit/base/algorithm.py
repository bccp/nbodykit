from nbodykit.extern.six import add_metaclass

import abc
import numpy
import logging

class Result(object):
    def __init__(self, alg):
        self.attrs = alg.attrs
        self.logger = alg.logger
        self.comm = alg.comm

    @property
    def state(self):
        d = {}
        d.update(self.__dict__)
        d.pop('logger')
        d.pop('comm')
        return d

    def save(self, output):
        """
        Save the results to the specified output file, as a pickle

        Parameters
        ----------
        output : str
            the string specifying the file to save
        """
        # only the master rank writes
        if self.comm.rank == 0:
            import pickle

            self.logger.info('measurement done; saving result to %s' % output)

            with open(output, 'wb') as ff:
                pickle.dump(self.state, ff) 

@add_metaclass(abc.ABCMeta)
class Algorithm(object):
    """
    Base class for an algorithm.

    An algorithm usually works on some data source and produces some data (result).

    The result can be either identical or striped with in the MPI comm.

    """
    logger = logging.getLogger('Algorithm')

    # called by the subclasses
    def __init__(self, comm):

        # ensure self.comm is set, though usually already set by the child.

        self.comm = comm

    @property
    def attrs(self):
        """
        Dictionary storing relevant meta-data
        """
        try:
            return self._attrs
        except AttributeError:
            self._attrs = {}
            return self._attrs

    @property
    def results(self):
        """
            result of the algorithm
        """
        try:
            return self._results
        except AttributeError:
            self._results = Result(self)
            return self._results
        
    @abc.abstractmethod
    def run(self):
        """ Shall return None"""
        pass

