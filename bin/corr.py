import numpy
import logging
from mpi4py import MPI
import os

rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()
logging.basicConfig(level=logging.DEBUG,
                    format='rank %d on %s: '%(rank,name) + \
                            '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('corr.py')
              
from nbodykit import measurestats
from nbodykit.plugins import ArgumentParser
from nbodykit.extensionpoints import DataSource
from nbodykit.extensionpoints import MeasurementStorage

def binning_type(s):
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
        
        
    

def initialize_parser(**kwargs):
    parser = ArgumentParser("Brutal Correlation Function Calculator",**kwargs)

    # add the positional arguments
    parser.add_argument("mode", choices=["1d", "2d"]) 
    parser.add_argument("output", help='write correlation function to this file. set as `-` for stdout') 
    parser.add_argument("rbins", type=binning_type, help='the string specifying the binning to use') 


    # add the input field types
    h = "one or two input fields, specified as:\n\n"
    parser.add_argument("inputs", nargs="+", type=DataSource.create, 
                        help=h+DataSource.format_help())

    # add the optional arguments
    parser.add_argument("--subsample", type=int, default=1,
                        help='Use 1 out of every N points')
    parser.add_argument("--los", choices="xyz", default='z',
                        help="the line-of-sight direction, which the angle `mu` is defined with respect to")
    parser.add_argument("--Nmu", type=int, default=10,
                        help='the number of mu bins to use (from mu=-1 to mu=1) -- only used if `mode == 2d`')
    parser.add_argument('--poles', type=lambda s: [int(i) for i in s.split()], default=[],
                        metavar="0 2 4",
                        help='if specified, compute the multipoles for these `ell` values from xi(r,mu)')
    return parser 


def compute_brutal_corr(ns, comm=None):
    """
    Compute the correlation function via direct pair summation and save
    the results
    """
    if comm is None: comm = MPI.COMM_WORLD
    
    # check multipoles parameters
    if len(ns.poles) and ns.mode == '2d':
        raise ValueError("you specified multipole numbers but `mode` is `2d` -- perhaps you meant `1d`")
    
    # set Nmu to 1 if doing 1d
    if ns.mode == "1d": ns.Nmu = 0
    
    # call the function
    kw = {'comm':comm, 'subsample':ns.subsample, 'Nmu':ns.Nmu, 'los':ns.los, 'poles':ns.poles}
    pc, xi, RR = measurestats.compute_brutal_corr(ns.inputs, ns.rbins, **kw)

    # output
    if comm.rank == 0:
        storage = MeasurementStorage.new(ns.mode, ns.output)
        
        if ns.mode == '1d':
            if len(ns.poles):
                cols = ['r'] + ['corr_%d' %l for l in ns.poles] + ['RR', 'N']
                result = [pc.mean_centers] + [xi[:,i] for i in range(len(ns.poles))] + [RR, pc.pair_counts]
            else:
                cols = ['r', 'corr', 'RR', 'N']
                result = [pc.mean_centers, xi, RR, pc.pair_counts]
            edges = pc.edges[0]
        else:
            cols = ['r', 'mu', 'corr', 'RR', 'N']
            r, mu = pc.mean_centers
            result = [r, mu, xi, RR, pc.pair_counts]
            edges = pc.edges

        storage.write(edges, cols, result)
        logger.info('brutal corr done')

def main():
    
    # parse
    ns = initialize_parser().parse_args()
    
    # do the work
    compute_brutal_corr(ns)
    
if __name__ == '__main__':
    main()
