from abc import abstractmethod, abstractproperty
import numpy
import logging

from pmesh import window
from pmesh.pm import RealField, ComplexField

class Painter(object):
    """
    Painter object to help Sources convert results from Source.read to a RealField.

    The real field shall have a normalization of real.value = 1 + delta = n / nbar.
    """
    logger = logging.getLogger("Painter")
    def __init__(self, paintbrush='cic', interlaced=False):
        """
        Parameters
        ----------
        paintbrush : str, optional
            the string specifying the interpolation kernel to use when gridding the discrete field to the mesh
        interlaced : bool, optional
            whether to use interlacing to minimize aliasing effects
        """
        self.paintbrush = paintbrush
        self.interlaced = interlaced

    def __call__(self, source, pm):
