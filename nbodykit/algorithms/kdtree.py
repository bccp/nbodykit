import os
import numpy
import logging

from pmesh.domain import GridND
from scipy.spatial.ckdtree import cKDTree as KDTree

def split_size_3d(s):
    """ Split `s` into three integers, 
        a, b, c, such that a * b * c == s and a <= b <= c

        returns:  a, d
    """
    a = int(s** 0.3333333) + 1
    d = s
    while a > 1:
        if s % a == 0:
            s = s // a
            break
        a = a - 1 
    b = int(s ** 0.5) + 1
    while b > 1:
        if s % b == 0:
            s = s // b
            break
        b = b - 1
    c = s
    return a, b, c

class KDDensity(object):
    """ Estimates a proxy density based on the distance to the nearest neighbour.
        The result is proportional to the density but the scale is unspecified.
    """
    logger = logging.getLogger('KDDensity')
    def __init__(self, source, margin=1.0):
        """
            Parameters
            ----------
            margin: float
            Padding region per domain; relative to the mean seperation

        """
        self.comm = source.comm
        self._source = source

        BoxSize = numpy.array(self._source.attrs['BoxSize'], dtype='f8')
        if BoxSize.ndim == 0 or len(BoxSize) == 1: # catch some common problems about BoxSize.
            BoxSize = numpy.array([BoxSize, BoxSize, BoxSize])

        self.attrs = {}
        self.attrs['BoxSize'] = BoxSize
        self.attrs['meansep'] = (self._source.csize / BoxSize.prod()) ** (1.0 / len(BoxSize))
        self.attrs['margin'] = margin
        self.run()

    def run(self):
        Np = split_size_3d(self.comm.size)
        edges = [numpy.linspace(0, self.attrs['BoxSize'][d], Np[d] + 1, endpoint=True) for d in range(3)]
        domain = GridND(comm=self.comm, periodic=True, edges=edges)

        # read all position.
        pos = self._source.compute(self._source['Position'])

        layout = domain.decompose(pos, smoothing=self.attrs['margin'] * self.attrs['meansep'])

        xpos = layout.exchange(pos)
        # wait for scipy 0.19.1
        assert all(self.attrs['BoxSize'] == self.attrs['BoxSize'][0])
        xpos[...] /= self.attrs['BoxSize']
        xpos %= 1

        tree = KDTree(xpos, boxsize=1.0)
        d, i = tree.query(xpos, k=[8])
        d = d[:, 0]
        d = layout.gather(d, mode=numpy.fmin)
        self.density = 1 / (d ** 3 * self.attrs['BoxSize'].prod())


