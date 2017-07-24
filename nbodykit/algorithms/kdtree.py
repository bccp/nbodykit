import os
import numpy
import logging

from nbodykit.utils import split_size_3d
from pmesh.domain import GridND
from scipy.spatial.ckdtree import cKDTree as KDTree

class KDDensity(object):
    """
    Estimate a proxy density based on the distance to the nearest neighbor.
    The result is proportional to the density but the scale is unspecified.

    Results are computed when the object is inititalized. See the documenation
    of :func:`~KDDensity.run` for the attributes storing the results.

    Parameters
    ----------
    source : CatalogSource
        the input source of particles to compute the proxy density on;
        must specify the 'Position' column
    margin: float, optional
        Padding region per parallel domain; relative to the mean seperation
    """
    logger = logging.getLogger('KDDensity')

    def __init__(self, source, margin=1.0):

        if 'Position' not in source:
            raise ValueError("please specify the 'Position' column in the input source")

        self.comm = source.comm
        self._source = source

        if 'BoxSize' not in self._source.attrs:
            raise ValueError("please specify 'BoxSize' in the input source 'attrs'")
        BoxSize = numpy.array(self._source.attrs['BoxSize'], dtype='f8')
        if BoxSize.ndim == 0 or len(BoxSize) == 1: # catch some common problems about BoxSize.
            BoxSize = numpy.array([BoxSize, BoxSize, BoxSize])

        # store some meta-data
        self.attrs = {}
        self.attrs['BoxSize'] = BoxSize
        self.attrs['meansep'] = (self._source.csize / BoxSize.prod()) ** (1.0 / len(BoxSize))
        self.attrs['margin'] = margin

        # run the algorithm
        self.run()

    def run(self):
        """
        Compute the density proxy. This attaches the following attribute:

        - :attr:`density`

        Attributes
        ----------
        density : array_like, length: :attr:`size`
            a unit-less, proxy density value for each object on the local
            rank. This is computed as the inverse cube of the distance
            to the closest, nearest neighbor
        """

        # do the domain decomposition
        Np = split_size_3d(self.comm.size)
        edges = [numpy.linspace(0, self.attrs['BoxSize'][d], Np[d] + 1, endpoint=True) for d in range(3)]
        domain = GridND(comm=self.comm, periodic=True, edges=edges)

        # read all position and exchange
        pos = self._source.compute(self._source['Position'])
        layout = domain.decompose(pos, smoothing=self.attrs['margin'] * self.attrs['meansep'])
        xpos = layout.exchange(pos)

        # wait for scipy 0.19.1
        assert all(self.attrs['BoxSize'] == self.attrs['BoxSize'][0])
        xpos[...] /= self.attrs['BoxSize']
        xpos %= 1

        # KDTree
        tree = KDTree(xpos, boxsize=1.0)
        d, i = tree.query(xpos, k=[8])
        d = d[:, 0]

        # gather back to original root, taking the minimum distance
        d = layout.gather(d, mode=numpy.fmin)
        self.density = 1 / (d ** 3 * self.attrs['BoxSize'].prod())
