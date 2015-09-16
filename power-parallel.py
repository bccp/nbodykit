from mpi4py import MPI
import argparse
import os
import tempfile
import logging
from power import initialize_power_parser, compute_power

# Initializations and preliminaries
comm = MPI.COMM_WORLD 
size = comm.size
rank = comm.rank
name = MPI.Get_processor_name()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

from nbodykit.utils.mpilogging import MPILoggerAdapter

logger = MPILoggerAdapter(logging.getLogger('power-parallel.py'))
                    
#------------------------------------------------------------------------------
# tools
#------------------------------------------------------------------------------
def iter_values(s):
    """
    Parse the counter values, first trying to eval, 
    and then trying to just split the string by spaces into 
    a list
    """
    casters = [lambda s: eval(s), lambda s: s.split()]
    for cast in casters:
        try:
            return cast(s)
        except Exception as e:
            continue
    else:
        raise RuntimeError("failure to parse iteration value string `%s`: %s" %(s, str(e)))
        
def extra_iter_values(s):
    """
    Provided an existing file name, read the file into 
    a dictionary. The keys are interpreted as the string format name, 
    and the values are a list of values to iterate over for each job
    """
    if not os.path.exists(s):
        raise RuntimeError("file `%s` does not exist" %s)
    toret = {}
    execfile(s, globals(), toret)
    return toret
        
def split_ranks(N_ranks, N_chunks):
    """
    Divide the ranks into N chunks, removing the master (0) rank
    
    Parameters
    ----------
    N_ranks : int
        the total number of ranks available
    N_chunks : int
        the number of chunks to split the ranks into
    """
    seq = range(1, N_ranks)
    avg = int((N_ranks-1) // N_chunks)
    remainder = (N_ranks-1) % N_chunks

    start = 0
    end = avg
    for i in range(N_chunks):
        if remainder:
            end += 1
            remainder -= 1
        yield i, seq[start:end]
        start = end
        end += avg
        
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
    
#------------------------------------------------------------------------------
# job running
#------------------------------------------------------------------------------
def run_single_task(task, itask, config, param_file, pool_comm, parser):
    """
    Run a single job, calling `nbodykit::power.py` with the parameters
    specified for this job iteration
    
    Parameters
    ----------
    task :  
        the value of the counter that specifies this iteration
    itask : int
        the integer index of this task
    config : argparse.Namespace
        the namespace with command-line options for `power-parallel.py`
    param_file : str
        the parameter file for `power.py` as a string, which will be
        string formatted
    pool_comm : MPI.Communicator
        the communicator for this pool of worker processes
    parser : argparse.Namespace
        the argument parser for `power.py`
    """
    pool_rank = pool_comm.rank

    # if you are the pool's root, write out the temporary parameter file
    temp_name = None
    if pool_rank == 0:
        # generate the parameter file
        kwargs = [kw for _, kw, _, _ in param_file._formatter_parser() if kw]
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            temp_name = ff.name
            possible_kwargs = {config.iter_str : task}
            if config.extras is not None:
                for k in config.extras:
                    possible_kwargs[k] = config.extras[task][itask]
            valid = {k:v for k,v in possible_kwargs.iteritems() if k in kwargs}
            ff.write(param_file.format(**valid))
    temp_name = pool_comm.bcast(temp_name, root=0)

    # parse the file
    ns = parser.parse_args(['@%s' %temp_name])

    # set logging level
    logger.setLevel(ns.log_level)
    
    # compute the power using the comm for this worker pool
    compute_power(ns, comm=pool_comm)

    # remove temporary files
    if pool_rank == 0:
        if os.path.exists(temp_name):
            os.remove(temp_name)
            
def run_all_tasks(config, power_parser):
    """
    Run all tasks by having the `master` process distribute the tasks
    to the individual pools of worker processes
    """
    # read the parameter file
    param_file = open(config.param_file, 'r').read()
    
    # define MPI message tags
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    status = MPI.Status()
    
    # crash if we only have one process or one node
    if size <= config.nodes:
        raise ValueError("need `nodes+1` available processors to run `power-parallel.py`")    
      
    # split the ranks
    pool_comm = None
    chain_ranks = []
    color = 0
    worker_count = 0
    for i, ranks in split_ranks(size, config.nodes):
        chain_ranks.append(ranks[0])
        if rank in ranks: color = i+1
        worker_count += len(ranks)
    if worker_count != size-1:
        args = (worker_count, size-1)
        raise RuntimeError("mismatch between worker count (%d) and spawned worker processes (%d)" %args)
    pool_comm = comm.Split(color, 0)
    
    # the tasks provided on the command line
    tasks = config.iter_vals
    num_tasks = len(tasks)
    
    # master distributes the tasks
    if rank == 0:
        
        # initialize
        task_index = 0
        num_workers = config.nodes
        closed_workers = 0
        
        # loop until all workers have finished with no more tasks
        logger.info("master starting with %d node workers with %d total tasks" %(num_workers, num_tasks))
        while closed_workers < num_workers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            
            # worker is ready, so send it a task
            if tag == tags.READY:
                if task_index < num_tasks:
                    comm.send(tasks[task_index], dest=source, tag=tags.START)
                    logger.debug("sending task `%s` to worker %d" %(str(tasks[task_index]), source))
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)
            elif tag == tags.DONE:
                results = data
                logger.debug("received result from worker %d" %source)
            elif tag == tags.EXIT:
                closed_workers += 1
                logger.debug("worker %d has exited, closed workers = %d" %(source, closed_workers))
    
    # worker processes wait and execute single jobs
    else:
        if pool_comm.rank == 0:
            logger.info("pool master rank is %d on %s with %d processes available" % (rank, name, pool_comm.size))
        while True:
            task = -1
            tag = -1
        
            # have the master rank of the pool ask for task and then broadcast
            if pool_comm.rank == 0:
                comm.send(None, dest=0, tag=tags.READY)
                task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
            task = pool_comm.bcast(task)
            tag = pool_comm.bcast(tag)
        
            if tag == tags.START:
                # Do the work here
                result = run_single_task(task, tasks.index(task), config, param_file, pool_comm, power_parser)
                pool_comm.Barrier() # wait for everyone
                if pool_comm.rank == 0:
                    comm.send(result, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                break

        pool_comm.Barrier()
        if pool_comm.rank == 0:
            comm.send(None, dest=0, tag=tags.EXIT)
    
    # free and exit
    logger.debug("rank %d process finished" %rank)
    comm.Barrier()
    if rank == 0:
        logger.info("master is finished; terminating")
        pool_comm.Free()
    
def main():
    """
    The main function to run the jobs
    """
    # parse
    desc = "run several nbodykit::power.py tasks, possibly across several nodes"
    parser = argparse.ArgumentParser(description=desc) 
          
    h = """the name of the template file holding parameters for `power.py`;
           the file should use string `format` syntax to indicate which variables
           will be updated for each task, i.e., the output file could be 
           specified in the file as 'output/pkmu_output_box{box}.dat'"""
    parser.add_argument('param_file', type=str, help=h)
    
    h = 'the number of independent nodes that will run jobs in parallel; must be less than nprocs'
    parser.add_argument('nodes', type=int, help=h)
    
    h = """replace occurences of this string in template parameter file, using
           string `format` syntax, with the value taken from `iter_vals`"""
    parser.add_argument('iter_str', type=str, help=h)
    
    h = """the values to iterate over for each of the tasks, which can be supplied 
           as an evaluatable string (i.e., 'range(10)') or a set of values (i.e., 'A' 'B' 'C' 'D')"""
    parser.add_argument('iter_vals', type=iter_values, help=h)
    
    h = """file providing extra string replaces, with lines of the form `tag = ['tag1', 'tag2']`;
            if the keys match keywords in the template param file, the file with be updated with
            the ith value for the ith task"""
    parser.add_argument('--extra', dest='extras', type=extra_iter_values, help=h)
    
    h = "set the logging output to debug, with lots more info printed"
    parser.add_argument('--debug', help=h, action="store_const", dest="log_level", 
                            const=logging.DEBUG, default=logging.INFO)
    config = parser.parse_args()
    
    # set the logging level
    logger.setLevel(config.log_level)
        
    # initialize power.py parser
    power_parser = initialize_power_parser(args=['@'+config.param_file], add_help=False)
    
    # run
    run_all_tasks(config, power_parser)
   
if __name__ == '__main__' :
    main()
    
    
