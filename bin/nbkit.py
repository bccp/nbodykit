#! python

import argparse
import logging

from mpi4py import MPI
from nbodykit.extensionpoints import Algorithm, algorithms
from nbodykit.extensionpoints import Algorithm, DataSource, Transfer, Painter
from nbodykit.plugins import ArgumentParser, ListPluginsAction

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
    
    valid_algorithms = list(vars(algorithms).keys())

    # initialize the main parser
    desc = "Invoke an `nbodykit` algorithm with the given parameters. \n\n"
    desc += "MPI usage: mpirun -n [n] python nbkit.py ... \n\n"
    desc += "Because MPI standard requires the python interpreter in mpirun commandline. \n"
    kwargs = {}
    kwargs['description'] = desc
    kwargs['fromfile_prefix_chars'] = '@'
    kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter

    parser = ArgumentParser("nbkit.py", add_help=False, **kwargs)

    parser.add_argument('-o', '--output', help='the string specifying the output')
    parser.add_argument('algorithm_name', choices=valid_algorithms)
    parser.add_argument('-h', '--help', action=HelpAction, help='Help on an algorithm')
    
    parser.add_argument('--list-datasources', action=ListPluginsAction(DataSource), help='List DataSource')
    parser.add_argument('--list-algorithms', action=ListPluginsAction(Algorithm), help='List Algorithms')
    parser.add_argument('--list-painters', action=ListPluginsAction(Painter), help='List Painters')
    parser.add_argument('--list-transfers', action=ListPluginsAction(Transfer), help='List Transfer Functions')

    parser.usage = parser.format_usage()[6:-1] + " ... \n"
    
    # parse the command-line
    ns, args = parser.parse_known_args()
    alg_name = ns.algorithm_name; output = ns.output

    # configuration file passed via -c
    if ns.config is not None:
        params, extra = Algorithm.parse_known_yaml(alg_name, ns.config)
        output = getattr(extra, 'output', None)
    
    # output is required
    if output is None:
        raise ValueError("argument -o/--output is required")
            
    # initialize the algorithm and run
    if ns.config is not None:
        alg_class = getattr(algorithms, alg_name)
        alg = alg_class(**vars(params))
    else:
        alg = Algorithm.create([alg_name]+args)
    
    # run and save
    result = alg.run()
    alg.save(output, result) 
       
if __name__ == '__main__':
    main()
