from nbodykit.io.stack import FileStack
from nbodykit.base.particles import ParticleSource
from nbodykit.base.painter import Painter

import numpy

class ZeldovichParticles(ParticleSource):
    """
    A source of particles Poisson-sampled from density fields in the Zel'dovich approximation
    """
    def __init__(self, comm, cosmo, nbar, redshift, BoxSize, Nmesh, bias=2., rsd=None, seed=None):
        
        # communicator and cosmology
        self.comm    = comm
        self.cosmo   = cosmo
        
        # save the meta-data
        self.attrs['nbar']     = nbar
        self.attrs['redshift'] = redshift
        self.attrs['BoxSize']  = BoxSize
        self.attrs['Nmesh']    = Nmesh
        self.attrs['bias']     = bias
        self.attrs['rsd']      = rsd
        self.attrs['seed']     = seed
        
        if self.comm.rank == 0:
            self.logger.info("attrs = %s" % self.attrs)
            
        # generate the data and set the collective size
        self._csize = self.comm.allreduce(len(self._source))
            
    def __getitem__(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        if col in self.transform:
            return self.transform[col](self)
        elif col in self._source.dtype.names:
            import dask.array as da
            return da.from_array(self._source[col], chunks=100000)
        else:
            raise KeyError("column `%s` is not a valid column name" %col)
        
    @property
    def size(self):
        return len(self._source)
        
    @property
    def csize(self):
        return self._csize
        
    @property
    def columns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        return sorted(set(list(self._source.dtype.names) + list(self.transform)))

    @property
    def _source(self):
        """
        The underlying data array which holds the `Position` data
        """
        try:
            return self._pos
        except AttributeError:
            
            # classylss is required to call CLASS and create a power spectrum
            try: import classylss
            except: raise ImportError("`classylss` is required to use %s" %self.plugin_name)
        
            # the other imports
            from nbodykit import mockmaker
            from pmesh.pm import ParticleMesh
            from nbodykit.utils import MPINumpyRNGContext
        
            # initialize the CLASS parameters 
            pars = classylss.ClassParams.from_astropy(self.cosmo)

            try:
                cosmo = classylss.Cosmology(pars)
            except Exception as e:
                raise ValueError("error running CLASS for the specified cosmology: %s" %str(e))
        
            # initialize the linear power spectrum object
            Plin = classylss.power.LinearPS(cosmo, z=self.attrs['redshift'])
        
            # the particle mesh for gridding purposes
            pm = ParticleMesh(BoxSize=self.attrs['BoxSize'], Nmesh=[self.attrs['Nmesh']]*3, dtype='f4', comm=self.comm)
        
            # generate initialize fields and Poisson sample with fixed local seed
            with MPINumpyRNGContext(self.attrs['seed'], self.comm):
        
                # compute the linear overdensity and displacement fields
                delta, disp = mockmaker.gaussian_real_fields(pm, Plin, compute_displacement=True)
        
                # sample to Poisson points
                f = cosmo.f_z(self.attrs['redshift']) # growth rate to do RSD in the Zel'dovich approx
                kws = {'rsd':self.attrs['rsd'], 'f':f, 'bias':self.attrs['bias']}
                pos = mockmaker.poisson_sample_to_points(delta, disp, pm, self.attrs['nbar'], **kws)
            
            # logging
            self.logger.debug("local number of Zeldovich particles = %d" %len(pos))
            size = self.comm.allreduce(len(pos))
            if self.comm.rank == 0:
                self.logger.info("total number of Zeldovich particles = %d" %size)
            
            dtype = numpy.dtype([('Position', (pos.dtype.str,3))])
            self._pos = numpy.empty(len(pos), dtype=dtype)
            self._pos['Position'][:] = pos[:]
            return self._pos
    
    def read(self, columns):
        """
        Return the requested columns as dask arrays
        
        Currently, this returns a dask array holding the total amount
        of data for each rank, divided equally amongst the available ranks
        """
        return [self[col] for col in columns]

    def paint(self, pm):
        """
        Paint to the mesh
        """
        # paint and apply any transformations to the real field
        real = self.painter(self, pm)
        self.painter.transform(self, real)
        
        return real
