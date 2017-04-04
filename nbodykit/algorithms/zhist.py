import numpy
import logging
from mpi4py import MPI

from nbodykit import CurrentMPIComm
from nbodykit.transform import ConstantArray

def scotts_bin_width(data, comm):
    """
    Return the optimal histogram bin width using Scott's rule, 
    defined as: 
    
    .. math:: h = \sigma \sqrt[3]{\frac{24 * \sqrt{\pi}}{n}}
    
    Notes
    -----
    This is a collective operation
    
    Parameters
    ----------
    data : array_like
        the array that we are histograming
    comm : 
        the MPI communicator
    
    Returns
    -------
    dx : float
        the bin spacing
    edges : array_like
        the array holding the bin edges
    """
    # compute the mean
    csum = comm.allreduce(data.sum())
    csize = comm.allreduce(data.size)
    cmean = csum / csize
    
    # std dev
    rsum = comm.allreduce((abs(data - cmean)**2).sum())
    sigma = (rsum / csize)**0.5

    dx = sigma * (24. * numpy.sqrt(numpy.pi) / csize) ** (1. / 3)
    maxval = comm.allreduce(data.max(), op=MPI.MAX)    
    minval = comm.allreduce(data.min(), op=MPI.MIN)    
    
    Nbins = numpy.ceil((maxval - minval) * 1. / dx)
    Nbins = max(1, Nbins)
    edges = minval + dx * numpy.arange(Nbins + 1)
    return dx, edges
    
class RedshiftHistogram(object):
    """
    Compute the mean number density as a function of redshift 
    :math:`n(z)` from an input Source of particles.
    
    .. warning:: The units of the number density are :math:`(\mathrm{Mpc}/h)^{-3}`
    """
    logger = logging.getLogger('RedshiftHistogram')

    def __init__(self, source, fsky, cosmo, bins=None, redshift='Redshift', weight=None):
        """
        Parameters
        ----------
        source : CatalogSource
            the source of particles holding the redshift column to histogram
        fsky : float
            the sky area fraction, which is used in the volume calculation when
            normalizing :math:`n(z)`
        cosmo : nbodykit.cosmology.Cosmology
            the cosmological parameters, which are used to compute the volume
            from redshift shells when normalizing :math:`n(z)`
        bins : int or sequence of scalars; optional
            If `bins` is an int, it defines the number of equal-width
            bins in the given range. If `bins` is a sequence, it defines the bin 
            edges, including the rightmost edge, allowing for non-uniform bin widths.
            If not provided, Scott's rule is used to estimate the optimal bin width
            from the input data (default)
        redshift : str; optional
            the name of the column specifying the redshift data
        weight : str; optional
            the name of the column specifying weights to use when histogramming the data
        """
        # input columns need to be there
        for col in [redshift, weight]:
            if col is not None and col not in source:
                raise ValueError("'%s' column missing from input source in RedshiftHistogram" %col)
        
        self.comm = source.comm
        
        # using Scott's rule for binning
        if bins is None:
            h, bins = scotts_bin_width(source.compute(source[redshift]), self.comm)
            if self.comm.rank == 0:
                self.logger.info("using Scott's rule to determine optimal binning; h = %.2e, N_bins = %d" %(h, len(bins)-1))
            
        # equally spaced bins from min to max val
        elif numpy.isscalar(bins):
            if self.comm.rank == 0:
                self.logger.info("computing %d equally spaced bins" %bins)
            z = source.compute(source[redshift])
            maxval = comm.allreduce(z.max(), op=MPI.MAX)    
            minval = comm.allreduce(z.min(), op=MPI.MIN)
            bins = linspace(minval, maxval, bins + 1, endpoint=True)
                        
        self.source = source
        self.cosmo  = cosmo
        
        self.attrs             = {}
        self.attrs['edges']    = bins
        self.attrs['fsky']     = fsky
        self.attrs['redshift'] = redshift
        self.attrs['weight']   = weight
        self.attrs.update({'cosmo.%s' %k:cosmo[k] for k in cosmo})
        
        # and run
        self.run()
                         
    def run(self):
        """
        Run the algorithm, which computes the histogram. This function
        does not return anything, but adds several attributes
        to the class (see below).
        
        Attributes
        ----------
        bin_edges : array_like
            the edges of the redshift bins
        bin_centers : array_like
            the center values of each redshift bin
        dV : array_like
            the volume of each redshift shell in units of :math:`(\mathrm{Mpc}/h)^3`
        nbar : array_like
            the values of the redshift histogram, normalized to 
            number density (in units of :math:`(\mathrm{Mpc}/h)^{-3}`)
        """       
        edges = self.attrs['edges']
         
        # get the columns
        redshift = self.source[self.attrs['redshift']]
        if self.attrs['weight'] is not None:
            weight = self.source[self.attrs['weight']]
            if self.comm.rank == 0:
                self.logger.info("computing histogram using weights from '%s' column" %self.attrs['weight'])
        else:
            weight = ConstantArray(1.0, self.source.size)
            
        # compute the numpy arrays from dask
        redshift, weight = self.source.compute(redshift, weight)
        
        # do the bin count, using the specified weight values
        dig = numpy.searchsorted(edges, redshift, "right")
        N = numpy.bincount(dig, weights=weight, minlength=len(edges)+1)[1:-1]
    
        # now sum across all ranks
        N = self.comm.allreduce(N)
    
        # compute the volume
        if self.comm.rank == 0:
            self.logger.info("using cosmology %s to compute volume in units of (Mpc/h)^3" %str(self.cosmo))
            self.logger.info("sky fraction used in volume calculation: %.4f" %self.attrs['fsky'])
        R_hi = self.cosmo.comoving_distance(edges[1:]).value * self.cosmo.h
        R_lo = self.cosmo.comoving_distance(edges[:-1]).value * self.cosmo.h
        dV   = (4./3.)*numpy.pi*(R_hi**3 - R_lo**3) * self.attrs['fsky']
    
        # store the results
        self.bin_edges   = edges
        self.bin_centers = 0.5*(edges[:-1] + edges[1:])
        self.dV          = dV
        self.nbar        = 1.*N/dV
        
    def __getstate__(self):
        state = dict(
                     bin_edges=self.bin_edges,
                     bin_centers=self.bin_centers,
                     dV=self.dV,
                     nbar=self.nbar,
                     attrs=self.attrs)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, output):
        """ 
        Save the RedshiftHistogram result to disk.

        The format is JSON.
        """
        import json
        from nbodykit.utils import JSONEncoder
        
        # only the master rank writes
        if self.comm.rank == 0:
            self.logger.info('histogram done; saving result to %s' %output)

            with open(output, 'w') as ff:
                json.dump(self.__getstate__(), ff, cls=JSONEncoder) 
            
    @classmethod
    @CurrentMPIComm.enable
    def load(cls, output, comm=None):
        """
        Load a saved RedshiftHistogram result.

        The result has been saved to disk with :func:`RedshiftHistogram.save`.
        """
        import json
        from nbodykit.utils import JSONDecoder
        if comm.rank == 0:
            with open(output, 'r') as ff:
                state = json.load(ff, cls=JSONDecoder)
        else:
            state = None
        state = comm.bcast(state)
        self = object.__new__(cls)
        self.__setstate__(state)
        self.comm = comm
        return self
