from nbodykit.extensionpoints import DataSource, Cosmology
from nbodykit.extensionpoints import datasources

from mpi4py import MPI
import numpy
import logging
from scipy.interpolate import InterpolatedUnivariateSpline as spline

logger = logging.getLogger('SurveyCatalog')

def nbar_fromfile(filename):
    """
    Read the number density from file, returning a spline of (z, nbar)
    """
    import os
    if not os.path.exists(filename):
        raise ValueError("'%s' file for reading nbar does not exist")
    
    # assumes two columns: (z, nbar)
    d = numpy.loadtxt(filename)
    return InterpolatedUnivariateSpline(d[:,0], d[:,1])
 
class TracerCatalogDataSource(DataSource):
    """
    A `DataSource` to represent a catalog of tracer objects, measured
    in an observational survey, with a non-trivial selection function. 
    
    The key attributes of a `TracerCatalog` are:
    
        * data: 
            a `RaDecRedshift` DataSource that reads the (ra, dec, z)
            of the true tracer objects, whose intrinsic clustering 
            is non-zero
        * randoms: 
            a `RaDecRedshift` DataSource that reads the (ra, dec, z)
            of a catalog of objects generated randomly to match the
            survey geometry and whose instrinsic clustering is zero
        * BoxSize:
            the size of the cartesian box -- the Cartesian coordinates
            of the input objects are computed using the input cosmology,
            and then placed into the box
        * offset: 
            the average coordinate value in each dimension -- this offset
            is used to return cartesian coordinates translated into the
            domain of [-BoxSize/2, BoxSize/2]
    """
    plugin_name = "TracerCatalog"
            
    @classmethod
    def register(kls):
        
        h = kls.parser
        
        # required arguments
        h.add_argument("data", type=datasources.RaDecRedshift.fromstring, 
            help="`RaDecRedshift` DataSource representing the `data` catalog")
        h.add_argument("randoms", type=datasources.RaDecRedshift.fromstring, 
            help="`RaDecRedshift` DataSource representing the `randoms` catalog")
        h.add_argument("cosmo", type=Cosmology.fromstring, 
            help='the cosmology used to convert (ra,dec,z) to cartesian coordinates')
            
        # optional arguments
        h.add_argument("-BoxSize", type=kls.BoxSizeParser,
            help="the size of the box; if not provided, automatically computed from the `randoms` catalog")
        h.add_argument('-BoxPad', type=float, default=0.02, 
            help='when setting the box size automatically, apply this additional buffer')
        h.add_argument('-compute_fkp_weights', action='store_true', 
            help='if set, use FKP weights, computed from `P0_fkp` and the provided `nbar`') 
        h.add_argument('-P0_fkp', type=float,
            help='the fiducial power value `P0` used to compute FKP weights')
        h.add_argument('-nbar', type=nbar_fromfile,
            help='read `nbar(z)` from this file, which provides two columns (z, nbar)')
        h.add_argument('-fsky', type=float, 
            help='the sky area fraction of the tracer catalog, used in the volume calculation of `nbar`')
        
      
    def finalize_attributes(self):
        """
        Finalize the attributes by performing several steps:
        
            1. if `BoxSize` not provided on the command-line, 
               infer the value from the Cartesian coordinates of
               the `randoms` catalog
            2. compute the mean coordinate offset for each 
               Cartesian dimension -- used to re-center the 
               coordinates to the [-BoxSize/2, BoxSize/2] domain
            3. compute the number density as a function of redshift
               from the `randoms` and store a spline
        """
        # source is None by default
        self._source = None

        # sample the cosmology's comoving distance
        self.cosmo.sample('comoving_distance', numpy.logspace(-5, 1, 1024))
    
        # read the data to find total number
        data_stats = {}
        for result in self.data.read(['Position'], data_stats, full=False):
            continue
        N_data = data_stats['Ntot']
        
        redshifts = []
        randoms_stats = {}
        self.offset = None
        
        # need to compute cartesian min/max
        coords_min = numpy.array([numpy.inf]*3)
        coords_max = numpy.array([-numpy.inf]*3)
        
        # now loop over the randoms and determine min/max and get the redshifts
        for [coords] in self.data.read(['Position'], randoms_stats, full=False):
            
            # get the global min/max of cartesian
            if self.comm.rank == 0:
                cartesian = self._to_cartesian(coords)
                coords_min = numpy.minimum(coords_min, cartesian.min(axis=0))
                coords_max = numpy.maximum(coords_max, cartesian.max(axis=0))
                
                # store the redshifts
                redshifts += list(coords[:,-1])
        N_ran = randoms_stats['Ntot']
        
        # only rank zero does the work, then broadcast
        if self.comm.rank == 0:
            
            # setup the box, using randoms to define it
            self._define_box(coords_min, coords_max)
    
            # compute the number density from the randoms
            self._set_nbar(numpy.array(redshifts), alpha=1.*N_data/N_ran)
            
        # broadcast the results that rank 0 computed
        self.BoxSize   = self.comm.bcast(self.BoxSize)
        self.offset    = self.comm.bcast(self.offset)
        self.nbar      = self.comm.bcast(self.nbar)
        
        if self.comm.rank == 0:
            logger.info("BoxSize = %s" %str(self.BoxSize))
            logger.info("cartesian coordinate range: %s : %s" %(str(coords_min), str(coords_max)))
            logger.info("mean coordinate offset = %s" %str(self.offset))
        
    def _define_box(self, coords_min, coords_max):
        """
        Define the Cartesian box to hold the tracers by:
        
            * computing the Cartesian coordinates for all objects
            * setting the `BoxSize` attribute, if not provided
            * computing the coorindate offset needed to center the
              data onto the [-BoxSize/2, BoxSize/2] domain
        """   
        # center the data in the first cartesian quadrant
        delta = abs(coords_max - coords_min)
        self.offset = 0.5 * (coords_min + coords_max)
        
        # set the box size automatically
        if self.BoxSize is None:
            delta *= 1.0 + self.BoxPad
            self.BoxSize = delta.astype(int)
        else:
            # check the input size
            for i, L in enumerate(delta):
                if self.BoxSize[i] < L:
                    args = (self.BoxSize[i], i, L)
                    logger.warning("input BoxSize of %.2f in dimension %d smaller than coordinate range of data (%.2f)" %args)
                            
    def _to_cartesian(self, coords, translate=[0.,0.,0.]):
        """
        Convert the (ra, dec, z) coordinates to cartesian coordinates
         
            * uses `self.cosmo` to compute comoving distances
            * optionally, translate the cartesian grid by the vector `translate`
        """
        ra, dec, redshift = coords.T
        r = self.cosmo.comoving_distance(redshift)
        x = r*numpy.cos(ra)*numpy.cos(dec)
        y = r*numpy.sin(ra)*numpy.cos(dec)
        z = r*numpy.sin(dec)
        return numpy.vstack([x,y,z]).T + translate
        
    def _set_nbar(self, redshift, alpha=1.0):
        """
        Determine the spline used to compute `nbar`
        """
        # if spline already exists, do nothing
        if self.nbar is not None:
            return 
            
        if self.fsky is None:
            raise ValueError("please specify `fsky` to compute volume needed for `nbar`")
        
        def scotts_bin_width(data):
            """
            Return the optimal histogram bin width using Scott's rule
            """
            n = data.size
            sigma = numpy.std(data)
            dx = 3.5 * sigma * 1. / (n ** (1. / 3))
            
            Nbins = numpy.ceil((data.max() - data.min()) * 1. / dx)
            Nbins = max(1, Nbins)
            bins = data.min() + dx * numpy.arange(Nbins + 1)
            return dx, bins
        
        # do the histogram of N(z)
        dz, zbins = scotts_bin_width(redshift)
        dig = numpy.searchsorted(zbins, redshift, "right")
        N = numpy.bincount(dig, minlength=len(zbins)+1)[1:-1]
        
        # compute the volume
        R_hi = self.cosmo.comoving_distance(zbins[1:])
        R_lo = self.cosmo.comoving_distance(zbins[:-1])
        volume = (4./3.)*numpy.pi*(R_hi**3 - R_lo**3) * self.fsky
        
        # store the nbar 
        z_cen = 0.5*(zbins[:-1] + zbins[1:])
        self.nbar = spline(z_cen, alpha*N/volume)
            
    def set_source(self, which):
        """
        Set the `source` point to either `data` or `randoms`, such
        that when `readall` is called, the results for that
        source are returned
        
        Set to `None` by default to remind the user to set it
        """
        if which == 'data':
            self._source = self.data
        elif which == 'randoms':
            self._source = self.randoms
        else:
            raise NotImplementedError("'source' must be set to either `data` or `randoms`")
        
    def read(self, columns, stats, full=False):
        """
        Read data from `source` by calling the `readall` function
        """
        # need to know which source to return from
        if self._source is None:
            raise ValueError("set `source` attribute to `data` or `randoms` by calling `set_source`")
            
        # check valid columns
        valid = ['Position', 'Weight', 'Nbar']
        if any(col not in valid for col in columns):
            args = (self.__class__.__name__, str(valid))
            raise ValueError("valid `columns` to read from %s: %s" %args)
            
        # compute normalization A and shot noise S
        stats['A'] = 0.
        stats['S'] = 0.
        
        # read (ra,dec,z) and weights and convert to cartesian
        for [coords, weight] in self._source.read(['Position', 'Weight'], stats, full=full):
            
            if self.comm.rank == 0:
                # cartesian coordinates, removing the mean offset in each dimension
                pos = self._to_cartesian(coords, translate=-self.offset)
        
                # number density from redshift
                nbar = self.nbar(coords[:,-1])
        
                # update the weights with new FKP
                if self.compute_fkp_weights:
                    if self.P0_fkp is None:
                        raise ValueError("if 'compute_fkp_weights' is set, please specify a value for 'P0_fkp'")
                    weight = 1. / (1. + nbar*self.P0_fkp)
                    
                P = {}
                P['Position'] = pos
                P['Weight']   = weight
                P['Nbar']     = nbar
                
                data = [P[key] for key in columns]        
                shape_and_dtype = [(d.shape, d.dtype) for d in data]
        
                # see equations 13-15 of Beutler et al 2013
                A = (nbar*weight**2).sum()
                S = (weight**2).sum()
            else:
                shape_and_dtype = None
                A = None; S = None
                
            shape_and_dtype = self.comm.bcast(shape_and_dtype)
            stats['A'] += self.comm.bcast(A)
            stats['S'] += self.comm.bcast(S)

            if self.comm.rank != 0:
                data = [
                    numpy.empty(0, dtype=(dtype, shape[1:]))
                    for shape,dtype in shape_and_dtype
                ]
            yield data