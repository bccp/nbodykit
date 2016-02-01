import argparse
import logging

from mpi4py import MPI
from nbodykit.extensionpoints import Algorithm, algorithms

# configure the logging
rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()
logging.basicConfig(level=logging.DEBUG,
                    format='rank %d on %s: '%(rank,name) + \
                            '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

def main():
    
    # the names of the valid algorithms -- each will be a subcommand
    valid_algorithms = list(vars(algorithms).keys())
    
    # the main parser usage
    usage = "%(prog)s [-h] " + "{%s}" %(','.join(valid_algorithms)) + " ... \n" 
    usage += "\nFrom more help on each of the subcommands, type:\n\n"
    usage += "\n".join("%(prog)s " + k + " -h" for k in valid_algorithms)
    usage += "\n\n"
    
    # initialize the main parser
    desc = "the main `nbodykit` executable, designed to run a number of analysis algorithms"
    kwargs = {}
    kwargs['usage'] = usage
    kwargs['description'] = desc
    kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(**kwargs)
        
    # add the subparsers for each `Algorithm` plugin
    subparsers = parser.add_subparsers(dest='algorithm_name')
    for s in valid_algorithms:
        
        # copy the parser for each Algorithm
        subparser = subparsers.add_parser(s)
        subparser.__dict__ = getattr(algorithms, s).parser.__dict__
        
        # add an output string
        subparser.add_argument('-o', '--output', required=True, type=str, 
                                help='the string specifying the output')

    # parse
    ns = parser.parse_args()
        
    # initialize the algorithm and run
    alg = Algorithm.create(ns.algorithm_name, ns, MPI.COMM_WORLD)
    result = alg.run()
    
    # save the output
    alg.save(ns.output, result) 
    
    
if __name__ == '__main__':
    main()
