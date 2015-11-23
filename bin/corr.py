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
    parser.add_argument("output", help='write correlation function to this file. set as `-` for stdout') 
    parser.add_argument("rmax", type=float, help='max distance') 
    parser.add_argument("Nbins", type=int, help='number of bins') 

    # add the input field types
    h = "one or two input fields, specified as:\n\n"
    parser.add_argument("inputs", nargs="+", type=plugins.DataSource.open, 
                        help=h+plugins.DataSource.format_help())

    # add the optional arguments
    parser.add_argument("--subsample", type=int, default=1,
                        help='Use 1 out of every N points')
    return parser 


def compute_brutal_corr(ns, comm=None):
    """
    
    """
    if comm is None: comm = MPI.COMM_WORLD
    
    # call the function
    kw = {'comm':comm, 'subsample':ns.subsample}
    pc, xi, RR = measurestats.compute_brutal_3d_corr(ns.inputs, ns.rmax, ns.Nbins, **kw)
    
    
    if comm.rank == 0:
        storage = plugins.MeasurementStorage.new('1d', ns.output)
        storage.write(pc.edges, ['r', 'corr', 'RR'], (pc.centers, xi, RR))
        logger.info('brutal corr done')

def main():
    """
    The main function
    """
    # parse
    ns = initialize_parser().parse_args()
    
    # do the work
    compute_brutal_corr(ns)
    
if __name__ == '__main__':
    main()
