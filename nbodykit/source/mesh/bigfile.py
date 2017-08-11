from __future__ import absolute_import
# the future import is important. or in python 2.7 we try to
# import this module itself. Due to the unfortnate name conflict!

from nbodykit.base.mesh import MeshSource
from nbodykit import CurrentMPIComm
from nbodykit.utils import JSONDecoder
from bigfile import BigFileMPI
from pmesh.pm import ParticleMesh, ComplexField, RealField

import numpy
import json
from six import string_types

class BigFileMesh(MeshSource):
    """
    A MeshSource object that reads a mesh from disk using :mod:`bigfile`.

    This can read meshes that have been stored with the
    :func:`~nbodykit.base.mesh.MeshSource.save` function of MeshSource objects.

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
    def __repr__(self):
        import os
        return "BigFileMesh(file=%s)" % os.path.basename(self.path)

    @CurrentMPIComm.enable
    def __init__(self, path, dataset, comm=None, **kwargs):

        self.path    = path
        self.dataset = dataset
        self.comm    = comm

        # update the meta-data
        self.attrs.update(kwargs)
        with BigFileMPI(comm=self.comm, filename=path)[dataset] as ff:
            for key in ff.attrs:
                v = ff.attrs[key]
                if isinstance(v, string_types) and v.startswith('json://'):
                    self.attrs[key] = json.loads(v[7:], cls=JSONDecoder)
                else:
                    self.attrs[key] = numpy.squeeze(v)

            # fourier space or config space
            if ff.dtype.kind == 'c':
                self.isfourier = True
                if ff.dtype.itemsize == 16:
                    dtype = 'f8'
                else:
                    dtype = 'f4'
            else:
                self.isfourier = False
                if ff.dtype.itemsize == 8:
                    dtype = 'f8'
                else:
                    dtype = 'f4'

        # determine Nmesh
        if 'ndarray.shape' not in self.attrs:
            raise ValueError("`ndarray.shape` should be stored in the Bigfile `attrs` to determine `Nmesh`")

        if 'Nmesh' not in self.attrs:
            raise ValueError("`ndarray.shape` should be stored in the Bigfile `attrs` to determine `Nmesh`")

        Nmesh = self.attrs['Nmesh']
        BoxSize = self.attrs['BoxSize']

        MeshSource.__init__(self, BoxSize=BoxSize, Nmesh=Nmesh, dtype=dtype, comm=comm)

    def to_real_field(self):
        """
        Return the RealField stored on disk.

        .. note::
            The mesh stored on disk must be stored with ``mode=real``

        Returns
        -------
        real : pmesh.pm.RealField
            an array-like object holding the mesh loaded from disk in
            configuration space
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
        """
        Return the ComplexField stored on disk.

        .. note::
            The mesh stored on disk must be stored with ``mode=complex``

        Returns
        -------
        real : pmesh.pm.ComplexField
            an array-like object holding the mesh loaded from disk in Fourier
            space
        """
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
