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

        if self.paintbrush not in window.methods:
            raise ValueError("valid ``paintbrush`` values: %s" %str(window.methods))

    def __call__(self, source, pm):
        """
        Paint the input `source` to the mesh specified by `pm`
        
        Parameters
        ----------
        source : `Source` or a subclass
            the source object from which the default 
        pm : pmesh.pm.ParticleMesh
            the particle mesh object
        
        Returns
        -------
        real : pmesh.pm.RealField
            the painted real field
        """
        Nlocal = 0 # number of particles read on local rank
        
        # the paint brush window
        paintbrush = window.methods[self.paintbrush]

        # initialize the RealField to returns
        real = RealField(pm)
        real[:] = 0

        # need 2nd field if interlacing
        if self.interlaced:
            real2 = RealField(pm)
            real2[...] = 0

        # read the necessary data (as dask arrays)
        columns = ['Position', 'Weight', 'Selection']
        if not all(col in source for col in columns):
            missing = set(columns) - set(source.columns)
            raise ValueError("source does not contain columns: %s" %str(missing))
        Position, Weight, Selection = source.read(columns)

        # ensure the slices are synced, since decomposition is collective
        N = max(pm.comm.allgather(len(Position)))

        # paint data in chunks on each rank
        chunksize = 1024 ** 2
        for i in range(0, N, chunksize):
            if i > len(Position) : i = len(Position)
            s = slice(i, i + chunksize)
            position, weight, selection = source.compute(Position[s], Weight[s], Selection[s])

            if weight is None:
                weight = numpy.ones(len(position))

            if selection is not None:
                position = position[selection]
                weight   = weight[selection]

            Nlocal += len(position)

            if not self.interlaced:
                lay = pm.decompose(position, smoothing=0.5 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)
                real.paint(position, mass=weight, method=paintbrush, hold=True)
            else:
                lay = pm.decompose(position, smoothing=1.0 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)

                H = pm.BoxSize / pm.Nmesh

                # in mesh units
                shifted = pm.affine.shift(0.5)

                real.paint(position, mass=weight, method=paintbrush, hold=True)
                real2.paint(position, mass=weight, method=paintbrush, transform=shifted, hold=True)
                c1 = real.r2c()
                c2 = real2.r2c()

                for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                    kH = sum(k[i] * H[i] for i in range(3))
                    s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

                c1.c2r(real)
        nbar = pm.comm.allreduce(Nlocal) / numpy.prod(pm.BoxSize)

        if nbar > 0:
            real[...] /= nbar

        real.shotnoise = 1 / nbar

        if pm.comm.rank == 0:
            self.logger.info("mean number density is %g", nbar)
            self.logger.info("normalized the convention to 1 + delta")

        return real
