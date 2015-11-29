import logging
from mpi4py import MPI

from corr import initialize_parser, compute_brutal_corr
from nbodykit.utils.taskmanager import TaskManager

# setup the logging
rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()
logging.basicConfig(level=logging.DEBUG,
                    format='rank %d on %s: '%(rank,name) + \
                            '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('corr-parallel.py')


def main():
    
    # parse
    desc = "run several nbodykit::corr.py tasks, possibly across several nodes"
    config = TaskManager.parse_args(desc)
    
    # set the logging level
    logger.setLevel(config.log_level)
            
    # initialize power.py parser
    task_parser = initialize_parser(args=['@'+config.param_file], add_help=False)
    
    # initialize manager and run all
    manager = TaskManager(compute_brutal_corr, config, task_parser)
    manager.run_all()
   
if __name__ == '__main__' :
    main()
    
    