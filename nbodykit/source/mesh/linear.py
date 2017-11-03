from nbodykit.base.mesh import MeshSource
from nbodykit import CurrentMPIComm, mockmaker
from nbodykit.utils import attrs_to_dict
import numpy

class LinearMesh(MeshSource):
    """
    A MeshSource object that generates a :class:`~pmesh.pm.RealField` density
    mesh from a linear power spectrum function :math:`P(k)`.

    Parameters
    ----------
    Plin: callable
        the callable linear power spectrum function, which takes the
        wavenumber as its single argument
    BoxSize : float, 3-vector of floats
        the size of the box to generate the grid on
    Nmesh : int, 3-vector of int
        the number of the mesh cells per side
    seed : int, optional
        the global random seed, used to set the seeds across all ranks
    remove_variance : bool, optional
        .. deprecated:: 0.2.9
            use ``unitary_amplitude`` instead
    unitary_amplitude: bool, optional
        ``True`` to remove variance from the complex field by fixing the
        amplitude to :math:`P(k)` and only the phase is random.
    inverted_phase: bool, optional
        ``True`` to invert phase of the complex field by fixing the
        amplitude to :math:`P(k)` and only the phase is random.
    comm : MPI communicator
        the MPI communicator
    """
    def __repr__(self):
        return "LinearMesh(seed=%(seed)d)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self, Plin, BoxSize, Nmesh, seed=None,
            unitary_amplitude=False,
            inverted_phase=False,
            remove_variance=None,
            comm=None):

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
        if remove_variance is not None:
            unitary_amplitude = remove_variance

        self.attrs['unitary_amplitude'] = unitary_amplitude
        self.attrs['inverted_phase'] = inverted_phase

        MeshSource.__init__(self, BoxSize=BoxSize, Nmesh=Nmesh, dtype='f4', comm=comm)

    def to_complex_field(self):
        """
        Return a ComplexField, generating from the linear power spectrum.

        .. note::

            The density field is normalized to :math:`1+\delta` such that
            the mean of the return field in real space is unity.

        Returns
        -------
        :class:`pmesh.pm.ComplexField`
            an array-like object holding the generated linear density
            field in Fourier space
        """
        # generate linear density field with desired seed
        complex, _ = mockmaker.gaussian_complex_fields(self.pm, self.Plin, self.attrs['seed'],
                    unitary_amplitude=self.attrs['unitary_amplitude'],
                    inverted_phase=self.attrs['inverted_phase'],
                    compute_displacement=False)
        # set normalization to 1 + \delta.
        def filter(k, v):
            mask = numpy.bitwise_and.reduce([ki == 0 for ki in k])
            v[mask] = 1.0
            return v

        complex.apply(filter)
        complex.attrs = {}
        complex.attrs.update(self.attrs)
        return complex
