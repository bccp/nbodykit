from nbodykit.io.stack import FileStack
from nbodykit.base.particles import ParticleSource
from nbodykit.base.painter import Painter

from nbodykit import CurrentMPIComm

import numpy

class ZeldovichParticles(ParticleSource):
    """
    A source of particles Poisson-sampled from density fields in the Zel'dovich approximation
    """
    @CurrentMPIComm.enable
    def __init__(self, cosmo, nbar, redshift, BoxSize, Nmesh, bias=2., rsd=None, seed=None, comm=None):
        
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

        ParticleSource.__init__(self, comm)

    def __getitem__(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        if col in self._source.dtype.names:
            import dask.array as da
            return da.from_array(self._source[col], chunks=100000)

        return ParticleSource.__getitem__(self, col)

    @property
    def size(self):
        return len(self._source)

    @property
    def hcolumns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        return list(self._source.dtype.names)

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
            
            dtype = numpy.dtype([('Position', (pos.dtype.str,3))])
            self._pos = numpy.empty(len(pos), dtype=dtype)
            self._pos['Position'][:] = pos[:]
            return self._pos
