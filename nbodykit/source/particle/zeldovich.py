from nbodykit.base.particles import ParticleSource
from nbodykit import cosmology
from nbodykit.utils import attrs_to_dict
from nbodykit import CurrentMPIComm

import numpy

class ZeldovichParticles(ParticleSource):
    """
    A source of particles Poisson-sampled from density fields in the Zel'dovich approximation
    """
    def __repr__(self):
        return "ZeldovichParticles(seed=%(seed)d, bias=%(bias)g)" % self.attrs

    @CurrentMPIComm.enable
    def __init__(self, Plin, nbar, BoxSize, Nmesh, bias=2., rsd=None, seed=None, comm=None, cosmo=None, redshift=None):
        """
        Parameters
        ----------
        Plin : callable
            callable specifying the linear power spectrum
        nbar : float
            the number density of the particles in the box, assumed constant across the box; 
            this is used when Poisson sampling the density field
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
            the global random seed; if set to ``None``, the seed will be set randomly
        comm : MPI communicator, optional
            the MPI communicator
        cosmo : nbodykit.cosmology.Cosmology, optional
            this must be supplied if `Plin` does not carry ``cosmo`` attribute, and 
            we wish to apply RSD, so that the growth rate f(z) can be computed
        redshift : float, optional
            this must be supplied if `Plin` does not carry a ``redshift`` attribute, and 
            we wish to apply RSD, so that the growth rate f(z) can be computed
        """
        self.comm = comm
        self.Plin = Plin
        
        if rsd is None:
            rsd = [0, 0, 0.]
        
        # must be passed only if we wish to do RSD
        if cosmo is None:
            cosmo = getattr(self.Plin, 'cosmo', None)
        if redshift is None:
            redshift = getattr(self.Plin, 'redshift', None)
        if sum(rsd) and cosmo is None or redshift is None:
            raise ValueError("if RSD is requested, ``redshift`` and ``cosmo`` keywords must be passed")
        self.cosmo = cosmo
        
        # try to add attrs from the Plin
        if isinstance(Plin, cosmology.LinearPowerBase):
            self.attrs.update(Plin.attrs)
        else:
            self.attrs.update({'cosmo.%s' %k:cosmo[k] for k in cosmo})

        # save the meta-data
        self.attrs['nbar']     = nbar
        self.attrs['redshift'] = redshift
        self.attrs['bias']     = bias
        self.attrs['rsd']      = rsd
        
        # set the seed randomly if it is None
        if seed is None:
            if self.comm.rank == 0:
                seed = numpy.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs['seed'] = seed

        # init the base class
        ParticleSource.__init__(self, comm=comm)

        # make the actual source
        self._source, pm = self._makesource(BoxSize=BoxSize, Nmesh=Nmesh)
        self.attrs['Nmesh'] = pm.Nmesh.copy()
        self.attrs['BoxSize'] = pm.BoxSize.copy()

        # recompute _csize to the real size
        self.update_csize()
        
        # crash with no particles!
        if self.csize == 0:
            raise ValueError("no particles in ZeldovichParticles; try increasing ``nbar`` parameter")

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
        
        from nbodykit import mockmaker
        from pmesh.pm import ParticleMesh

        # the particle mesh for gridding purposes
        _Nmesh = numpy.empty(3, dtype='i8')
        _Nmesh[:] = Nmesh
        pm = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh, dtype='f4', comm=self.comm)

        # sample to Poisson points
        f = self.cosmo.growth_rate(self.attrs['redshift']) # growth rate to do RSD in the Zel'dovich approx

        # compute the linear overdensity and displacement fields
        delta, disp = mockmaker.gaussian_real_fields(pm, self.Plin, self.attrs['seed'], compute_displacement=True)

        # poisson sample to points
        kws = {'f':f, 'bias':self.attrs['bias'], 'seed':self.attrs['seed'], 'comm':self.comm}
        pos, vel = mockmaker.poisson_sample_to_points(delta, disp, pm, self.attrs['nbar'], **kws)

        # add RSD?
        pos += vel * self.attrs['rsd']

        # return data
        dtype = numpy.dtype([
                ('Position', ('f4', 3)),
                ('Velocity', ('f4', 3)),
        ])
        source = numpy.empty(len(pos), dtype)
        source['Position'][:] = pos[:]
        source['Velocity'][:] = vel[:]

        return source, pm
