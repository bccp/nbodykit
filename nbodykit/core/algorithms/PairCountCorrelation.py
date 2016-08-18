from nbodykit.core import Algorithm, DataSource
import numpy
import os

def binning_type(s):
    """
    Type conversion for use on the command-line that converts 
    a string to an array of bin edges
    """
    if os.path.isfile(s):
        return numpy.loadtxt(s)
    else:
        supported = ["`linspace: min max Nbins`", "`logspace: logmin logmax Nbins`"]
        try:
            f, params = s.split(':')
            params = list(map(float, params.split()))
            params[-1] = int(params[-1]) + 1

            if not hasattr(numpy, f): raise Exception
            if len(params) != 3: raise Exception

            return getattr(numpy, f)(*params)
        except:
            raise TypeError("supported binning format: [ %s ]" %", ".join(supported))


class PairCountCorrelationAlgorithm(Algorithm):
    """
    Algorithm to compute the 1d or 2d correlation function and/or multipoles
    via direct pair counting
    
    Notes
    -----
    The algorithm saves the correlation function result to a plaintext file, 
    as well as the meta-data associted with the algorithm. The names of the
    columns saved to file are:
    
        - r : 
            the mean separation in each `r` bin
        - mu : 2D corr only
            the mean value for each `mu` bin
        - corr : 
            the correlation function value
        - corr_X :
            the `X` multipole of the correlation function
        - RR : 
            the number of random-random pairs in each bin; used to 
            properly normalize the correlation function
        - N : 
            the number of pairs averaged over in each bin to compute
            the correlation function
    """
    plugin_name = "PairCountCorrelation"

    def __init__(self, mode, rbins, field, other=None, subsample=1, 
                    los='z', Nmu=10, poles=[]):
                    
        # positional arguments
        self.mode      = mode
        self.rbins     = rbins
        self.field     = field
        
        # keyword arguments
        self.other     = other
        self.subsample = subsample
        self.los       = los
        self.Nmu       = Nmu
        self.poles     = poles
        
        # construct the input fields list
        self.inputs = [self.field]
        if self.other is not None:
            self.inputs.append(self.other)
        
    @classmethod
    def fill_schema(cls):  
        s = cls.schema
        s.description = "correlation function calculator via pair counting"
    
        # the positional arguments
        s.add_argument("mode", type=str, choices=["1d", "2d"],
            help='measure the correlation function in `1d` or `2d`') 
        s.add_argument("rbins", type=binning_type, 
            help='the string specifying the binning to use') 
        s.add_argument("field", type=DataSource.from_config, 
            help='the first `DataSource` of objects to correlate; '
                 'run `nbkit.py --list-datasources` for all options')
        s.add_argument("other", type=DataSource.from_config, 
            help='the other `DataSource` of objects to cross-correlate with; '
                 'run `nbkit.py --list-datasources` for all options')
        s.add_argument("subsample", type=int, help='use 1 out of every N points')
        s.add_argument("los", choices="xyz",
            help="the line-of-sight: the angle `mu` is defined with respect to")
        s.add_argument("Nmu", type=int,
            help='if `mode == 2d`, the number of mu bins covering mu=[-1,1]')
        s.add_argument('poles', nargs='*', type=int,
            help='compute the multipoles for these `ell` values from xi(r,mu)')
   
    def run(self):
        """
        Run the pair-count correlation function and return the result
        
        Returns
        -------
        edges : list or array_like
            the array of 1d bin edges or a list of the bin edges in each dimension
        result : dict
            dictionary holding the data results (with associated names as keys) --
            this results `corr`, `RR`, `N` + the mean bin values
        """
        from nbodykit import measurestats
    
        # check multipoles parameters
        if len(self.poles) and self.mode == '2d':
            raise ValueError("you specified multipole numbers but `mode` is `2d` -- perhaps you meant `1d`")

        # set Nmu to 1 if doing 1d
        if self.mode == "1d": self.Nmu = 0

        # do the work
        kw = {'comm':self.comm, 'subsample':self.subsample, 'Nmu':self.Nmu, 'los':self.los, 'poles':self.poles}
        pc, xi, RR = measurestats.compute_brutal_corr(self.inputs, self.rbins, **kw)

        # format the results
        if self.mode == '1d':
            if len(self.poles):
                cols = ['r'] + ['corr_%d' %l for l in self.poles] + ['RR', 'N']
                result = [pc.mean_centers] + [xi[:,i] for i in range(len(self.poles))] + [RR, pc.pair_counts]
            else:
                cols = ['r', 'corr', 'RR', 'N']
                result = [pc.mean_centers, xi, RR, pc.pair_counts]
        else:
            cols = ['r', 'mu', 'corr', 'RR', 'N']
            r, mu = pc.mean_centers
            result = [r, mu, xi, RR, pc.pair_counts]

        return pc.edges, dict(zip(cols, result))
        
    def save(self, output, result):
        """
        Save the result returned by `run()` to the filename specified by `output`
        
        Parameters
        ----------
        output : str
            the string specifying the file to save
        result : tuple
            the tuple returned by `run()` -- first argument specifies the bin
            edges and the second is a dictionary holding the data results
        """
        from nbodykit.storage import MeasurementStorage
        
        # only master writes
        if self.comm.rank == 0:
            
            edges, result = result
            storage = MeasurementStorage.create(self.mode, output)
        
            cols = list(result.keys())
            values = list(result.values())
            storage.write(edges, cols, values)

            



