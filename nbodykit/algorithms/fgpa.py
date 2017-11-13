import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.dataset import DataSet
from nbodykit.meshtools import SlabIterator
from pmesh.pm import ComplexField

class FGPA(object):
    logger = logging.getLogger('FGPA')

    def __init__(self, source, A, gamma, Nlines, seed):
        self.tau_red = numpy.zeros((80, 10))
        field = source.paint(mode='real')
        
        self.logger.info(field.cnorm())

    def save(self, output):
        pass

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, output, comm=None):
        pass
