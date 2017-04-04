from nbodykit.base.catalog import CatalogSource, column
from nbodykit import cosmology
from nbodykit.utils import attrs_to_dict
from nbodykit import CurrentMPIComm

import numpy

class LogNormalCatalog(CatalogSource):
    """
    A catalog source containing (biased) particles that have 
    been Poisson-sampled from a log-normal density field
    """
    def __repr__(self):
        return "LogNormalCatalog(seed=%(seed)d, bias=%(bias)g)" %self.attrs

    @CurrentMPIComm.enable
    def __init__(self, Plin, nbar, BoxSize, Nmesh, bias=2., seed=None,
                    cosmo=None, redshift=None, comm=None, use_cache=False):
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
        seed : int, optional
            the global random seed; if set to ``None``, the seed will be set randomly
        cosmo : nbodykit.cosmology.Cosmology, optional
            this must be supplied if `Plin` does not carry ``cosmo`` attribute
        redshift : float, optional
            this must be supplied if `Plin` does not carry a ``redshift`` attribute
        comm : MPI Communicator, optional
            the MPI communicator instance; default (``None``) sets to the
            current communicator  
        use_cache : bool, optional
            whether to cache data read from disk; default is ``False``
        """
        self.comm = comm
        self.Plin = Plin
        
        # try to infer cosmo or redshift from Plin
        if cosmo is None:
            cosmo = getattr(self.Plin, 'cosmo', None)
        if redshift is None:
            redshift = getattr(self.Plin, 'redshift', None)
        if cosmo is None:
            raise ValueError("'cosmo' must be passed if 'Plin' does not have 'cosmo' attribute")
        if redshift is None:
            raise ValueError("'redshift' must be passed if 'Plin' does not have 'redshift' attribute")
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
        
        # set the seed randomly if it is None
        if seed is None:
            if self.comm.rank == 0:
                seed = numpy.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs['seed'] = seed

        # init the base class
        CatalogSource.__init__(self, comm=comm, use_cache=use_cache)

        # make the actual source
        self._source, pm = self._makesource(BoxSize=BoxSize, Nmesh=Nmesh)
        self.attrs['Nmesh'] = pm.Nmesh.copy()
        self.attrs['BoxSize'] = pm.BoxSize.copy()

        # recompute _csize to the real size
        self.update_csize()
        
        # crash with no particles!
        if self.csize == 0:
            raise ValueError("no particles in LogNormal source; try increasing ``nbar`` parameter")

    @property
    def size(self):
        if not hasattr(self, "_source"):
            return NotImplemented
        return len(self._source)

    @column
    def Position(self):
        """
        Position assumed to be in Mpc/h
        """
        return self.make_column(self._source['Position'])

    @column
    def Velocity(self):
        """
        Velocity in km/s
        """
        return self.make_column(self._source['Velocity'])
        
    @column
    def VelocityOffset(self):
        """
        The corresponding RSD offset, in Mpc/h
        """
        return self.make_column(self._source['VelocityOffset'])

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
        # this returns position and velocity offsets
        kws = {'bias':self.attrs['bias'], 'seed':self.attrs['seed'], 'comm':self.comm}
        pos, disp = mockmaker.poisson_sample_to_points(delta, disp, pm, self.attrs['nbar'], **kws)

        # move particles from initial position based on the Zeldovich displacement
        pos[:] = (pos + disp) % BoxSize
                
        # RSD in the Zel'dovich approx bring in extra factor of f
        # add this to both velocity and velocity offset
        disp[:] *= (1 + f)
        
        # velocity from displacement (assuming Mpc/h)
        # this is f * H(z) * a / h = f 100 E(z) a --> converts from Mpc/h to km/s
        z = self.attrs['redshift']
        velocity_norm = f * 100 * self.cosmo.efunc(z) / (1+z)
        vel = velocity_norm * disp

        # return data
        dtype = numpy.dtype([
                ('Position', ('f4', 3)),
                ('Velocity', ('f4', 3)),
                ('VelocityOffset', ('f4', 3))
        ])
        source = numpy.empty(len(pos), dtype)
        source['Position'][:] = pos[:] # in Mpc/h
        source['Velocity'][:] = vel[:] # in km/s
        source['VelocityOffset'][:] = disp[:] # in Mpc/h
        
        return source, pm
