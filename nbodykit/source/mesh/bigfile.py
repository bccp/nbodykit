from __future__ import absolute_import
# the future import is important. or in python 2.7 we try to 
# import this module itself. Due to the unfortnate name conflict!

from nbodykit.base.mesh import MeshSource
from nbodykit import CurrentMPIComm
from bigfile import BigFileMPI
import numpy

from pmesh.pm import ParticleMesh, ComplexField, RealField

class BigFileMesh(MeshSource):
    """
    Read a grid that was stored on disk using :mod:`bigfile`
    
    This can read grids stored using the ``PaintGrid`` algorithm
    """
    @CurrentMPIComm.enable
    def __init__(self, path, dataset, comm=None, **kwargs):
        """
        Parameters
        ----------
        path : str
            the name of the file to load
        dataset : str
            the name of the dataset in the Bigfile holding the grid
        comm : MPI.Communicator
            the MPI communicator
        **kwargs : 
            extra meta-data to be stored in the :attr:`attrs` dict
        """
        self.path    = path
        self.dataset = dataset
        self.comm    = comm

        # update the meta-data
        self.attrs.update(kwargs)
        with BigFileMPI(comm=self.comm, filename=path)[dataset] as ff:
            for key in ff.attrs:
                self.attrs[key] = numpy.squeeze(ff.attrs[key])

            # fourier space or config space
            if ff.dtype.kind == 'c':
                self.isfourier = True
            else:
                self.isfourier = False

        # determine Nmesh
        if 'ndarray.shape' not in self.attrs:
            raise ValueError("`ndarray.shape` should be stored in the Bigfile `attrs` to determine `Nmesh`")

        if 'Nmesh' not in self.attrs:
            raise ValueError("`ndarray.shape` should be stored in the Bigfile `attrs` to determine `Nmesh`")

        Nmesh = self.attrs['Nmesh']
        BoxSize = self.attrs['BoxSize']

        MeshSource.__init__(self, BoxSize=BoxSize, Nmesh=Nmesh, dtype='f4', comm=comm)
        
    def to_real_field(self):
        """
        Load a grid from file, and paint to the ParticleMesh represented by ``pm``
        
        Parameters
        ----------
        pm : pmesh.pm.ParticleMesh
            the particle mesh object to which we will paint the grid
        
        Returns
        -------
        real : pmesh.pm.RealField
            an array-like object holding the interpolated grid
        """
        
        if self.isfourier:
            return NotImplemented

        # the real field to paint to
        pmread = self.pm

        with BigFileMPI(comm=self.comm, filename=self.path)[self.dataset] as ds:
            if self.comm.rank == 0:
                self.logger.info("reading real field from %s" % self.path)
            real2 = RealField(pmread)
            start = sum(self.comm.allgather(real2.size)[:self.comm.rank])
            end = start + real2.size
            real2.unsort(ds[start:end])

        return real2

    def to_complex_field(self):
        if not self.isfourier:
            return NotImplemented
        pmread = self.pm

        if self.comm.rank == 0:
            self.logger.info("reading complex field from %s" % self.path)

        with BigFileMPI(comm=self.comm, filename=self.path)[self.dataset] as ds:
            complex2 = ComplexField(pmread)
            assert self.comm.allreduce(complex2.size) == ds.size
            start = sum(self.comm.allgather(complex2.size)[:self.comm.rank])
            end = start + complex2.size
            complex2.unsort(ds[start:end])

        return complex2
