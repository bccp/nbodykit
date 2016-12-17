from nbodykit.base.mesh import MeshSource
from nbodykit import CurrentMPIComm, mockmaker
from nbodykit.utils import MPINumpyRNGContext

class LinearMesh(MeshSource):
    """
    A source to generate a ``RealField`` grid directly from the 
    linear power spectrum.
    
    The linear power spectrum is computed using :mod:`classylss`, 
    which is a python wrapper around CLASS 
    """
    def __repr__(self):
        return "LinearMesh(seed=%(seed)d)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self, Plin, BoxSize, Nmesh, seed=None, comm=None):
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

        if hasattr(Plin, 'attrs'):
            for k in Plin.attrs:
                self.attrs['plin.' + k] = Plin.attrs[k]

        # save the rest of the attributes as meta-data
        self.attrs['seed']     = seed

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
        # generate linear density field with desired seed
        with MPINumpyRNGContext(self.attrs['seed'], self.comm):
            real, _ = mockmaker.gaussian_real_fields(self.pm, self.Plin, compute_displacement=False)

        return real

