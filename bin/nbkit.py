#! /usr/bin/env python

import argparse
import logging
import os, sys
from mpi4py import MPI

from nbodykit.extensionpoints import Algorithm, algorithms
from nbodykit.extensionpoints import DataSource, Transfer, Painter
from nbodykit.pluginmanager import ListPluginsAction, load

# configure the logging
rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()

def setup_logging(log_level):
    """
    Set the basic configuration of all loggers
    """
    logging.basicConfig(level=log_level,
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
        if rank == 0:
            if name is not None:
                print(Algorithm.format_help(name))
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
    parser = argparse.ArgumentParser("nbkit.py", add_help=False, **kwargs)

    # add the arguments to the parser
    parser.add_argument('algorithm_name', choices=valid_algorithms)
    parser.add_argument('config', type=argparse.FileType(mode='r'), nargs='?', default=None,
                        help='the name of the file to read parameters from using YAML syntax; '
                             'if not provided, stdin is read from')
    parser.add_argument('-h', '--help', action=HelpAction, help='help for a specific algorithm')
    parser.add_argument("-X", type=load, action="append", help="add a directory or file to load plugins from")
    parser.add_argument('-v', '--verbose', action="store_const", dest="log_level", 
                        const=logging.DEBUG, default=logging.INFO, 
                        help="run in 'verbose' mode, with increased logging output")
    
    # help arguments
    parser.add_argument('--list-datasources', nargs='*', action=ListPluginsAction(DataSource), 
        metavar='DataSource', help='list DataSource options')
    parser.add_argument('--list-algorithms',  nargs='*', action=ListPluginsAction(Algorithm), 
        metavar='Algorithm', help='list Algorithm options')
    parser.add_argument('--list-painters',  nargs='*', action=ListPluginsAction(Painter), 
        metavar='Painter', help='list Painter options')
    parser.add_argument('--list-transfers',  nargs='*', action=ListPluginsAction(Transfer), 
        metavar='Transfer', help='list Transfer options')

    # configure printing
    parser.usage = parser.format_usage()[6:-1] + " ... \n"
    if MPI.COMM_WORLD.rank != 0:
        parser._print_message = lambda x, file=None: None

    # parse the command-line
    ns, args = parser.parse_known_args()
    alg_name = ns.algorithm_name
    
    # setup logging
    setup_logging(ns.log_level)
        
    # configuration file passed via -c
    if ns.config is not None:
        # ns.config is a file object
        stream = ns.config.read()
    else:
        if MPI.COMM_WORLD.rank == 0:
            stream = sys.stdin.read()
        else:
            stream = None
        # respect the root rank stdin only;
        # on some systems, the stdin is only redirected to the root rank.
        stream = MPI.COMM_WORLD.bcast(stream)
    
    
    # expand environment variables in the input stream
    stream = os.path.expandvars(stream)
    params, extra = Algorithm.parse_known_yaml(alg_name, stream)
        
    # output is required
    if not hasattr(extra, 'output'):
        raise ValueError("parameter `output` is required in configuration; set to `None` for stdout")
    extra = vars(extra)
    output = extra.pop('output')
    
    # print warning if extra parameters ignored
    if MPI.COMM_WORLD.rank == 0 and len(extra):
        ignored = "[ %s ]" % ", ".join(["'%s'" %k for k in extra.keys()])
        logging.warning("the following keywords to `nbkit.py` have been ignored: %s" %ignored)
            
    # initialize the algorithm and run
    alg_class = getattr(algorithms, alg_name)
    alg = alg_class(**vars(params))

    # run and save
    result = alg.run()
    alg.save(output, result) 
       
if __name__ == '__main__':
    main()
