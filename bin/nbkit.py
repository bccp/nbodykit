#! /usr/bin/env python

import argparse
import logging
import os, sys
from mpi4py import MPI

from nbodykit import plugin_manager
from nbodykit.plugins import ListPluginsAction, EmptyConfigurationError
from nbodykit.plugins.fromfile import ReadConfigFile

# configure the logging
def setup_logging(log_level):
    """
    Set the basic configuration of all loggers
    """

    # This gives:
    #
    # [ 000000.43 ]   0:waterfall 06-28 14:49  measurestats    INFO     Nproc = [2, 1, 1]
    # [ 000000.43 ]   0:waterfall 06-28 14:49  measurestats    INFO     Rmax = 120

    import time
    logger = logging.getLogger();
    t0 = time.time()

    rank = MPI.COMM_WORLD.rank
    name = MPI.Get_processor_name().split('.')[0]

    class Formatter(logging.Formatter):
        def format(self, record):
            s1 = ('[ %09.2f ] % 3d:%s ' % (time.time() - t0, rank, name))
            return s1 + logging.Formatter.format(self, record)

    fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M ')

    hdlr = logging.StreamHandler()
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    logger.setLevel(log_level)

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
        if MPI.COMM_WORLD.rank != 0:
            parser.exit()
        
        if name is not None:
            parser.exit(0, plugin_manager.format_help('Algorithm', name))
        else:
            parser.exit(0, parser.format_help())
        
def main():
    
    # list of valid Algorithm plugin names
    valid_algorithms = list(plugin_manager['Algorithm'])

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
    parser.add_argument("-X", type=plugin_manager.add_user_plugin, action="append", 
                        help="add a directory or file to load plugins from")
    parser.add_argument('-v', '--verbose', action="store_const", dest="log_level", 
                        const=logging.DEBUG, default=logging.INFO, 
                        help="run in 'verbose' mode, with increased logging output")
    
    # add help arguments for extensions
    for extension in ['DataSource', 'Algorithm', 'Painter', 'Transfer']:
        arg = '--list-%ss' %extension.lower()
        parser.add_argument(arg, nargs='*', action=ListPluginsAction(extension, MPI.COMM_WORLD), 
                        metavar=extension, help='list help messages for %s plugins' %extension)

    # configure printing
    parser.usage = parser.format_usage()[6:-1] + " ... \n"
    if MPI.COMM_WORLD.rank != 0:
        parser._print_message = lambda x, file=None: None

    # parse the command-line
    ns, args = parser.parse_known_args()
    alg_name = ns.algorithm_name
    
    # setup logging
    setup_logging(ns.log_level)
        
    # configuration file parsing
    have_config_file = ns.config is not None
    if have_config_file:
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
    
    # parse the configuration file
    # print a useful message when no valid configuration was found
    try:
        alg_class = plugin_manager['Algorithm'][alg_name]
        params, extra = ReadConfigFile(stream, alg_class.schema)
    except EmptyConfigurationError:
        raise EmptyConfigurationError(("no configuration present; the user has two options for specifying configuration:\n"
                                       "\t1) pass the name of the configuration file as the second positional argument\n"
                                       "\t2) do not pass a second positional argument and the code will read the configuration from STDIN\n"
                                       "run \"python bin/nbkit.py -h\" for further details"))
        
    # output is required
    if not hasattr(extra, 'output'):
        raise ValueError("parameter `output` is required in configuration; set to `None` for stdout")
    extra = vars(extra)
    output = extra.pop('output')
    
    # print warning if extra parameters ignored
    if MPI.COMM_WORLD.rank == 0 and len(extra):
        ignored = "[ %s ]" % ", ".join(["'%s'" %k for k in extra.keys()])
        logging.warning("the following keywords to `nbkit.py` have been ignored: %s" %ignored)
            
    # initialize the algorithm with params from file
    alg = alg_class(**vars(params))

    # run and save
    result = alg.run()
    alg.save(output, result) 
       
import dask
dask.set_options(get=dask.get)
if __name__ == '__main__':
    main()
