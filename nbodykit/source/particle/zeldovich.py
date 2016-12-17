from nbodykit.base.particles import ParticleSource
from nbodykit.utils import attrs_to_dict, cosmology_to_dict
from nbodykit import CurrentMPIComm

import numpy

class ZeldovichParticles(ParticleSource):
    """
    A source of particles Poisson-sampled from density fields in the Zel'dovich approximation
    """
    def __repr__(self):
        return "ZeldovichParticles(seed=%(seed)d, bias=%(bias)g)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self,cosmo, nbar, redshift, BoxSize, Nmesh, Plin, bias=2., rsd=None, seed=None, comm=None):
        """
        Parameters
        ----------
        cosmo : cosmology
            the cosmology XXX: fixme.
        plin : callable
            linear power spectrum
        nbar : float
            the number density of the particles in the box, assumed constant across the box; 
            this is used when Poisson sampling the density field
        redshift : float
            the redshift of the linear power spectrum to generate
        BoxSize : float, 3-vector of floats
            the size of the box to generate the grid on
        Nmesh : int
            the mesh size to use when generating the density and displacement fields, which
            are Poisson-sampled to particles
        bias : float, optional
            the desired bias of the particles; applied while applying a log-normal transformation
            to the density field
        rsd : array_like, optional
            apply redshift space-distortions in this direction; if None, no rsd is applied.
        seed : int, optional
            the global random seed, used to set the seeds across all ranks
        comm : MPI communicator
            the MPI communicator
        """
        # communicator and cosmology
        self.comm    = comm
        self.cosmo = cosmo
        # FIXME: after cosmo can do power spectrum we shall 
        # use that as the default.

        self.Plin = Plin

        if rsd is None:
            rsd = [0, 0, 0.]

        self.attrs.update(attrs_to_dict(Plin, 'plin.'))
        self.attrs.update(cosmology_to_dict(cosmo, 'cosmo.'))

        # save the meta-data
        self.attrs['nbar']     = nbar
        self.attrs['redshift'] = redshift
        self.attrs['bias']     = bias
        self.attrs['rsd']      = rsd
        self.attrs['seed']     = seed

        ParticleSource.__init__(self, comm=comm)

        self._source, pm = self._makesource(BoxSize=BoxSize, Nmesh=Nmesh)

        self.attrs['Nmesh'] = pm.Nmesh.copy()
        self.attrs['BoxSize'] = pm.BoxSize.copy()

        # recompute _csize to the real size
        self.update_csize()

    def get_column(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        import dask.array as da
        return da.from_array(self._source[col], chunks=100000)

    @property
    def size(self):
        if not hasattr(self, "_source"):
            return NotImplemented
        return len(self._source)

    @property
    def hcolumns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        return list(self._source.dtype.names)

    def _makesource(self, BoxSize, Nmesh):

        # the other imports
        from nbodykit import mockmaker
        from pmesh.pm import ParticleMesh
        from nbodykit.utils import MPINumpyRNGContext

        # initialize the linear power spectrum object
        Plin = self.Plin

        # initialize the CLASS parameters 
        # FIXME: replace with our cosmology class
        import classylss
        pars = classylss.ClassParams.from_astropy(self.cosmo)
        try:
            cosmo = classylss.Cosmology(pars)
        except Exception as e:
            raise ValueError("error running CLASS for the specified cosmology: %s" %str(e))

        # the particle mesh for gridding purposes
        _Nmesh = numpy.empty(3, dtype='i8')
        _Nmesh[:] = Nmesh
        pm = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh, dtype='f4', comm=self.comm)

        # sample to Poisson points
        f = cosmo.f_z(self.attrs['redshift']) # growth rate to do RSD in the Zel'dovich approx

        # generate initialize fields and Poisson sample with fixed local seed
        with MPINumpyRNGContext(self.attrs['seed'], self.comm):
            # compute the linear overdensity and displacement fields
            delta, disp = mockmaker.gaussian_real_fields(pm, Plin, compute_displacement=True)

            kws = {'f':f, 'bias':self.attrs['bias']}
            pos, vel = mockmaker.poisson_sample_to_points(delta, disp, pm, self.attrs['nbar'], **kws)

        pos += vel * self.attrs['rsd']

        dtype = numpy.dtype([
                ('Position', ('f4', 3)),
                ('Velocity', ('f4', 3)),
        ])

        source = numpy.empty(len(pos), dtype)
        source['Position'][:] = pos[:]
        source['Velocity'][:] = vel[:]

        return source, pm
