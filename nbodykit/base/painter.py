from abc import abstractmethod, abstractproperty
import numpy
import logging

from pmesh import window
from pmesh.pm import RealField, ComplexField

class Painter(object):
    """
    Painter object to help Sources convert results from Source.read to a RealField 
    """
    def __init__(self, frho=None, fk=None, normalize=False, set_mean=None, paintbrush='cic', interlaced=False):
        """
        Parameters
        ----------
        frho : callable, optional
            a function for transforming the real-space density field; variables: (rho,)  
            example: 1 + (rho - 1)**2
        fk : callable, optional
            a function for transforming the Fourier-space density field; variables: (k, kx, ky, kz). 
            example: exp(-(k * 0.5)**2); applied before ``frho``
        normalize : bool, optional
            normalize the real-space field such that the mean is unity; applied before ``fk``
        set_mean : None, optional
            set the mean of the real-space field; applied after ``normalize``
        paintbrush : str, optional
            the string specifying the interpolation kernel to use when gridding the discrete field to the mesh
        interlaced : bool, optional
            whether to use interlacing to minimize aliasing effects
        """
        self.frho       = frho
        self.fk         = fk
        self.normalize  = normalize
        self.set_mean   = set_mean
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

        # size of the data the local rank is responsible for
        N = len(Position)
        
        # paint data in chunks on each rank
        chunksize = 1024 ** 2
        for i in range(0, N, chunksize):

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
                real.paint(position, mass=weight, method=paintbrush)
            else:
                lay = pm.decompose(position, smoothing=1.0 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)

                shifted = pm.affine.shift(shift)

                real.paint(position, mass=weight, method=paintbrush)
                real2.paint(position, mass=weight, method=paintbrush, transform=shifted)
                c1 = real.r2c()
                c2 = real2.r2c()

                H = pm.BoxSize / pm.Nmesh
                for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                    kH = sum(k[i] * H[i] for i in range(3))
                    s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

                c1.c2r(real)

        real.shotnoise = numpy.prod(pm.BoxSize) / pm.comm.allreduce(Nlocal)
        return real

    def transform(self, source, real):
        """
        Apply (in-place) transformations to the real-space field 
        specified by `real`
        """
        comm = real.pm.comm
        logger = source.logger

        # mean of the field
        mean = real.cmean()
        if comm.rank == 0: logger.info("Mean = %g" % mean)

        # normalize the field by dividing out the mean
        if self.normalize:
            real[...] *= 1. / mean
            mean = real.cmean()
            if comm.rank == 0: logger.info("Renormalized mean = %g" % mean)

        # explicity set the mean
        if self.set_mean is not None:
            real[...] += (self.set_mean - mean)

        # apply transformation in Fourier space
        if self.fk:
            
            if comm.rank == 0:
                logger.info("applying transformation fk %s" % self.fk)

            complex = real.r2c()
            for kk, slab in zip(complex.slabs.x, complex.slabs):
                k = sum([k ** 2 for k in kk]) ** 0.5
                slab[...] *= self.fk(k, kk[0], kk[1], kk[2])
            
            complex.c2r(real)
            mean = real.cmean()
            if comm.rank == 0:
                logger.info("after fk, mean = %g" % mean)
        
        # apply transformation in real-space
        if self.frho:
            if comm.rank == 0:
                logger.info("applying transformation frho %s" % self.frho)

            if comm.rank == 0:
                logger.info("example value before frho %g" % real.flat[0])
            for slab in real.slabs:
                slab[...] = self.frho(slab)
            if comm.rank == 0:
                logger.info("example value after frho %g" % real.flat[0])
            mean = real.cmean()
            if comm.rank == 0:
                logger.info("after frho, mean = %g" % mean)
