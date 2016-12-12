from nbodykit.base.grid import GridSource
from nbodykit.base.painter import Painter
from nbodykit import CurrentMPIComm
from bigfile import BigFileMPI
import numpy

class BigFileGrid(GridSource):
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
            self.attrs.update(ff.attrs)
            
            # fourier space or config space
            if ff.dtype.kind == 'c':
                self.isfourier = True
            else:
                self.isfourier = False

        # determine Nmesh
        if 'ndarray.shape' not in self.attrs:
            raise ValueError("`ndarray.shape` should be stored in the Bigfile `attrs` to determine `Nmesh`")
        self.Nmesh = self.attrs['ndarray.shape']

        # shot noise
        if 'shotnoise' in self.attrs:
            self.shotnoise = self.attrs['shotnoise'].squeeze()
        else:
            self.shotnoise = 0
            
        GridSource.__init__(self, comm)
        
    def paint(self, pm):
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
        from pmesh.pm import ParticleMesh, ComplexField, RealField
        
        # the real field to paint to
        real = RealField(pm)

        # check box size
        if not numpy.array_equal(pm.BoxSize, self.BoxSize):
            args = (self.BoxSize, pm.BoxSize)
            raise ValueError("`BoxSize` mismatch when painting grid: self.BoxSize = %s; pm.BoxSize = %s" %args)

        # the meshes do not match -- interpolation needed
        if any(pm.Nmesh != self.Nmesh):
            pmread = ParticleMesh(BoxSize=pm.BoxSize, Nmesh=self.Nmesh, dtype='f4', comm=self.comm)
        # no interpolation needed
        else:
            pmread = real.pm

        # open the dataset
        with BigFileMPI(comm=self.comm, filename=self.path)[self.dataset] as ds:

            # ComplexField
            if self.isfourier:
                if self.comm.rank == 0:
                    self.logger.info("reading complex field")
                complex2 = ComplexField(pmread)
                assert self.comm.allreduce(complex2.size) == ds.size
                start = sum(self.comm.allgather(complex2.size)[:self.comm.rank])
                end = start + complex2.size
                complex2.unsort(ds[start:end])
                complex2.resample(real)
            # RealField
            else:
                if self.comm.rank == 0:
                    self.logger.info("reading real field")
                real2 = RealField(pmread)
                start = sum(self.comm.allgather(real2.size)[:self.comm.rank])
                end = start + real2.size
                real2.unsort(ds[start:end])
                real2.resample(real)

        # pass on the shot noise
        real.shotnoise = self.shotnoise

        # apply transformations
        self.painter.transform(self, real)
        return real
