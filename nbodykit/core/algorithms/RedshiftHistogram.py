from nbodykit.core import Algorithm, DataSource
from nbodykit.distributedarray import GatherArray
import numpy

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

def RedshiftBins(bins):
    """
    The redshift bins to use when computing n(z)
    
    Parameters
    ----------
    bins : int or list of scalars
        If `bins` is a integer, it defines 
        the number of equal-width bins in the given range. If `bins` is a sequence,
        it defines the bin edges, including the rightmost edge, allowing
        for non-uniform bin widths
    """
    if isinstance(bins, list):
        bins = numpy.array(bins)
    elif isinstance(bins, str):
        if 'range' or 'arange':
            try:
                bins = eval(bins)
            except Exception as e:
                raise ValueError("error calling `eval` on `RedshiftBins` value: %s" %str(e))
    elif not isinstance(bins, int):
        raise ValueError("do not understand input ``RedshiftBins`` value of type %s" %type(bins))
    
    return bins
    

class RedshiftHistogramAlgorithm(Algorithm):
    """
    Algorithm to compute the mean number density of as a 
    function of redshift ``n(z)`` for the input ``DataSource``
    """
    plugin_name = "RedshiftHistogram"

    def __init__(self, datasource, bins=None, fsky=1.0, weight_col='Weight'):
        
        self.datasource = datasource
        self.bins       = bins
        self.fsky       = fsky
        self.weight_col = weight_col
        
        # set the cosmology
        self.cosmo = datasource.cosmo
        if self.cosmo is None:
            raise ValueError("`%s` algorithm requires a cosmology" %self.plugin_name)
        
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "compute n(z) from the input DataSource"
        
        s.add_argument("datasource", type=DataSource.from_config,
            help="DataSource with a `Redshift` column to compute n(z) from")
        
        s.add_argument('bins', type=RedshiftBins, 
            help=('the input redshift bins, specified as either as an integer or sequence of floats'))
        s.add_argument('fsky', type=float, 
            help='the sky area fraction, used in the volume calculation for `n(z)`')
        s.add_argument('weight_col', type=str, 
            help='the name of the column to use as a weight')
                                
    def run(self):
        """
        Run the algorithm, which returns (z, n(z))
        
        Returns
        -------
        zbins : array_like
            the redshift bin edges
        z_cen : array_like
            the center value of each redshift bin
        nz : array_like
            the n(z_cen) value
        """        
        # read the `Redshift` and `Weight` columns
        redshift = []; weights = []
        with self.datasource.open(defaults={self.weight_col:1.}) as stream:
            
            for [z, weight] in stream.read(['Redshift', self.weight_col], full=False):
                if len(z):
                    if not stream.isdefault('Redshift', z):
                        redshift += list(z)
                        weights += list(weight)
                    else:
                        raise DataSource.MissingColumn("no ``Redshift`` column in input DataSource")
        
        # gather to root and avoid MPI pickling limits
        redshift = GatherArray(numpy.array(redshift), self.comm, root=0)
        weights  = GatherArray(numpy.array(weights), self.comm, root=0)
        
        # root computes n(z)
        if self.comm.rank == 0:
            
            # use optimal bins via scotts bin width
            if self.bins is None:
                dz, zbins = scotts_bin_width(redshift)
            # use input bins
            elif isinstance(self.bins, int):
                zbins = numpy.linspace(redshift.min(), redshift.max(), self.bins+1, endpoint=True)
            else:
                zbins = self.bins
                
            # do the bin count, using the specified weight values
            dig = numpy.searchsorted(zbins, redshift, "right")
            N = numpy.bincount(dig, weights=weights, minlength=len(zbins)+1)[1:-1]
        
            # compute the volume
            R_hi = self.cosmo.comoving_distance(zbins[1:])
            R_lo = self.cosmo.comoving_distance(zbins[:-1])
            volume = (4./3.)*numpy.pi*(R_hi**3 - R_lo**3) * self.fsky
        
            # store the nbar 
            z_cen = 0.5*(zbins[:-1] + zbins[1:])
            nz = 1.*N/volume
        else:
            nz = None; z_cen = None; zbins = None
        
        zbins = self.comm.bcast(zbins)
        z_cen = self.comm.bcast(z_cen)    
        nz = self.comm.bcast(nz)
        
        return zbins, z_cen, nz

    def save(self, output, result):
        """
        Write (z_cen, n(z_cen)) to the specified file

        Parameters
        ----------
        output : str
            the string specifying the file to save
        result : tuple
            the tuple returned by `run()`, which holds (z, nz)
        """
        # only the master rank writes
        if self.comm.rank == 0:
            edges, z_cen, nz = result
            
            with open(output, 'wb') as ff:
                
                ff.write(("# z_min z_max z_cen n(z)\n").encode())
                out = numpy.vstack([edges[:-1], edges[1:], z_cen, nz]).T
                numpy.savetxt(ff, out, fmt='%.6e')
