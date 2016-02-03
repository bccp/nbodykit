import argparse
import logging

from mpi4py import MPI
from nbodykit.extensionpoints import Algorithm, algorithms
from nbodykit.plugins import ArgumentParser

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
    kwargs['fromfile_prefix_chars'] = '@'
    kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
    parser = ArgumentParser("nbkit", **kwargs)
        
    # add an output string
    parser.add_argument('-o', '--output', required=True, 
                            help='the string specifying the output')


    # add the subparsers for each `Algorithm` plugin
    parser.add_argument(dest='algorithm_name', choices=valid_algorithms)
    
    # parse
    ns, args = parser.parse_known_args()

    # initialize the algorithm and run
    alg = Algorithm.create([ns.algorithm_name] + args, MPI.COMM_WORLD)
    result = alg.run()
    
    # save the output
    alg.save(ns.output, result) 
    
    
if __name__ == '__main__':
    main()
