import numpy
import logging
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()
logging.basicConfig(level=logging.DEBUG,
                    format='rank %d on %s: '%(rank,name) + \
                            '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('corr.py')
              
from nbodykit import plugins, measurestats
from nbodykit.utils.pluginargparse import PluginArgumentParser

def initialize_parser(**kwargs):
    parser = PluginArgumentParser("Brutal Correlation unction Calculator",
                                    loader=plugins.load, **kwargs)

    # add the positional arguments
    parser.add_argument("mode", choices=["1d", "2d"]) 
    parser.add_argument("output", help='write correlation function to this file. set as `-` for stdout') 
    parser.add_argument("Rmax", type=float, help='the maximum radial distance') 
    parser.add_argument("Nr", type=int, help='the number of radial bins to use') 


    # add the input field types
    h = "one or two input fields, specified as:\n\n"
    parser.add_argument("inputs", nargs="+", type=plugins.DataSource.open, 
                        help=h+plugins.DataSource.format_help())

    # add the optional arguments
    parser.add_argument("--subsample", type=int, default=1,
                        help='Use 1 out of every N points')
    parser.add_argument("--los", choices="xyz", default='z',
                        help="the line-of-sight direction, which the angle `mu` is defined with respect to")
    parser.add_argument("--Nmu", type=int, default=10,
                        help='the number of mu bins to use (from mu=-1 to mu=1) -- only used if `mode == 2d`')
    parser.add_argument('--poles', type=lambda s: map(int, s.split()), default=[],
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
    pc, xi, RR = measurestats.compute_brutal_corr(ns.inputs, ns.Rmax, ns.Nr, **kw)

    # output
    if comm.rank == 0:
        storage = plugins.MeasurementStorage.new(ns.mode, ns.output)
        
        if ns.mode == '1d':
            if len(ns.poles):
                cols = ['r'] + ['corr_%d' %l for l in ns.poles] + ['RR', 'N']
                result = [pc.mean_centers] + [xi[:,i] for i in range(len(ns.poles))] + [RR, pc.pair_counts]
            else:
                cols = ['r', 'corr', 'RR', 'N']
                result = [pc.mean_centers, xi, RR, pc.pair_counts]
        else:
            cols = ['r', 'mu', 'corr', 'RR', 'N']
            r, mu = pc.mean_centers
            result = [r, mu, xi, RR, pc.pair_counts]

        storage.write(pc.edges, cols, result)
        logger.info('brutal corr done')

def main():
    
    # parse
    ns = initialize_parser().parse_args()
    
    # do the work
    compute_brutal_corr(ns)
    
if __name__ == '__main__':
    main()
