import logging
import warnings
from mpi4py import MPI
import numpy

rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()
logging.basicConfig(level=logging.DEBUG,
                    format='rank %d on %s: '%(rank,name) + \
                            '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('power.py')
              
import nbodykit
from nbodykit import plugins
from nbodykit.utils.pluginargparse import PluginArgumentParser
from kdcount import correlate

def initialize_parser(**kwargs):
    parser = PluginArgumentParser("Brutal Correlation function Calculator",
            loader=plugins.load,
            description=
         """
         """,
            epilog=
         """
         """,
            **kwargs
         )

    # add the positional arguments
    parser.add_argument("output", help='write correlation function to this file. set as `-` for stdout') 
    parser.add_argument("rmax", type=float, help='max distance') 
    parser.add_argument("Nbins", type=int, help='number of bins') 

    # add the input field types
    h = "one or two input fields, specified as:\n\n"
    parser.add_argument("inputs", nargs="+", type=plugins.DataSource.open, 
                        help=h+plugins.DataSource.format_help())

    # add the optional arguments
    parser.add_argument("--bunchsize", type=int, default=1024*1024*4,
        help='Number of particles to read per rank. A larger number usually means faster IO, but less memory for the FFT mesh. This is not respected by some data sources.')
    return parser 

def main():
    """
    The main function to initialize the parser and do the work
    """
    # parse
    ns = initialize_parser().parse_args()
    comm = MPI.COMM_WORLD
     
    [[pos1]] = ns.inputs[0].read(['Position'], comm, bunchsize=None)
    if len(ns.inputs) > 1:
        [[pos2]] = ns.inputs[1].read('Position', comm, bunchsize=None)
    else:
        pos2 = pos1

    pos1 = comm.gather(pos1)
    pos2 = comm.gather(pos2)
    if comm.rank == 0:
        pos1 = numpy.concatenate(pos1, axis=0)
        pos2 = numpy.concatenate(pos2, axis=0)
        pos2 = numpy.array_split(pos2, comm.size)

    pos1 = comm.bcast(pos1)
    pos2 = comm.scatter(pos2)

    tree1 = correlate.points(pos1, boxsize=ns.inputs[0].BoxSize)
    tree2 = correlate.points(pos2, boxsize=ns.inputs[0].BoxSize)

    if comm.rank == 0:
        logger.info('Rank 0 correlating %d x %d' % (len(tree1), len(tree2)))

    bins = correlate.RBinning(ns.rmax, ns.Nbins)
    pc = correlate.paircount(tree1, tree2, bins, np=0)
    pc.sum1[:] = comm.allreduce(pc.sum1)

    RR = 1.0 * len(pos1) * comm.allreduce(len(pos2)) 
    RR *= 4. / 3. * numpy.pi * numpy.diff(pc.edges**3/ ns.inputs[0].BoxSize.prod())

    xi = 1.0 * pc.sum1 / RR - 1
    
    if comm.rank == 0:
        storage = plugins.PowerSpectrumStorage.new('1d', ns.output)
        storage.write((pc.centers, xi, pc.sum1))

if __name__ == '__main__':
    main()
