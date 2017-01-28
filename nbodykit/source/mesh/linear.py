from nbodykit.base.mesh import MeshSource
from nbodykit import CurrentMPIComm, mockmaker
from nbodykit.utils import attrs_to_dict
import numpy

class LinearMesh(MeshSource):
    """
    A source to generate a ``RealField`` grid directly from a 
    linear power spectrum function
    """
    def __repr__(self):
        return "LinearMesh(seed=%(seed)d)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self, Plin, BoxSize, Nmesh, seed=None, remove_variance=False, comm=None):
        """
        Parameters
        ----------
        Plin: callable
            power spectrum of the field.
        BoxSize : float, 3-vector of floats
            the size of the box to generate the grid on
        Nmesh : int, 3-vector of int
            the number of the mesh points per side
        seed : int, optional
            the global random seed, used to set the seeds across all ranks
        comm : MPI communicator
            the MPI communicator
        """        
        self.Plin = Plin

        # cosmology and communicator
        self.comm    = comm

        self.attrs.update(attrs_to_dict(Plin, 'plin.'))

        # set the seed randomly if it is None
        if seed is None:
            if self.comm.rank == 0:
                seed = numpy.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs['seed'] = seed
        self.attrs['remove_variance'] = remove_variance

        MeshSource.__init__(self, BoxSize=BoxSize, Nmesh=Nmesh, dtype='f4', comm=comm)

    def to_complex_field(self):
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
        # generate linear density field with desired seed
        complex, _ = mockmaker.gaussian_complex_fields(self.pm, self.Plin, self.attrs['seed'], remove_variance=self.attrs['remove_variance'], compute_displacement=False)
        # set normalization to 1 + \delta.
        def filter(k, v):
            mask = numpy.bitwise_and.reduce([ki == 0 for ki in k])
            v[mask] = 1.0
            return v

        complex.apply(filter)
        complex.attrs = {}
        complex.attrs.update(self.attrs)
        return complex

