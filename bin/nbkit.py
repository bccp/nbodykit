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


class HelpAction(argparse.Action):
    """
    Help action that replicates the behavior of subcommands, 
    i.e., `python prog.py subcommand -h` prints the
    help for the subcommand
    """
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(HelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        name = getattr(namespace, 'algorithm_name')
        if name is not None:
            alg = getattr(algorithms, name)
            alg.parser.print_help()
        else:
            parser.print_help()
        parser.exit()
        
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
    kwargs['preparse_from_config'] = ['output']
    parser = ArgumentParser("nbkit", add_help=False, **kwargs)

        
    parser.add_argument('-o', '--output', help='the string specifying the output')
    parser.add_argument(dest='algorithm_name', choices=valid_algorithms)
    parser.add_argument('-h', '--help', action=HelpAction)
    
    # parse
    ns, args = parser.parse_known_args()
    if 'output' not in ns:
        raise ValueError("argument -o/--output is required")
    
    # initialize the algorithm and run
    alg = Algorithm.create([ns.algorithm_name] + args, MPI.COMM_WORLD)
    result = alg.run()
    
    # save the output
    alg.save(ns.output, result) 
       
if __name__ == '__main__':
    main()
