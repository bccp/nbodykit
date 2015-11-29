import logging
import os
import tempfile
from mpi4py import MPI

logger = logging.getLogger('taskmanager')

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
# task manager
#------------------------------------------------------------------------------
class TaskManager(object):
    """
    Task manager for running multiple power/correlation computations, 
    possibly in parallel using MPI
    """
    def __init__(self, task_function, config, task_parser, comm=None):
        """
        Parameters
        ----------
        task_function : callable
            the function to call for each task; arguments should be 
            an :py:class: `argparse.Namespace` and optionally
            a :py:class: `mpi4py.MPI.Intracomm` as the ``comm`` 
            keyword
        config : argparse.Namespace
            the namespace specifying the `TaskManager` configuration.
            see source code for ``TaskManager.parse_args`` for
            details on attributes
        task_parser : argparse.ArgumentParser
            the argument parser for the ``task_function`` that will 
            return the parameters to be passed to ``task_function``
            for each task
        comm : mpi4py.MPI.Intracomm, optional
            the global communicator, which will possibly be split
            and to use multiple comms across several nodes. If `None`,
            ``MPI.COMM_WORLD`` is used
        """
        self.task_function = task_function
        self.config        = config
        self.task_parser   = task_parser
        
        if comm is None: comm = MPI.COMM_WORLD
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        
    @classmethod
    def parse_args(cls, desc=None):
        """
        Parse command-line arguments that are needed to initialize a 
        `TaskManager` class
        
        Parameters
        ----------
        desc : str, optional
            the description of to use for this parser
        """
        import argparse
        
        # parse
        parser = argparse.ArgumentParser(description=desc) 
          
        h = """the name of the template file holding parameters needed to run the 
                desired task; the file should use ``string.format`` syntax to 
                indicate which variables will be updated for each task, i.e., the
                output file could be specified in the file as 
                'output/pkmu_output_box{box}.dat'"""
        parser.add_argument('param_file', type=str, help=h)
    
        h = """the number of independent nodes that will run jobs in parallel; 
                must be less than nprocs"""
        parser.add_argument('nodes', type=int, help=h)
    
        h = """replace occurences of this string in template parameter file, using
                ``string.format`` syntax, with the value taken from `iter_vals`"""
        parser.add_argument('iter_str', type=str, help=h)
    
        h = """the values to iterate over for each of the tasks, which can 
                be supplied as an evaluate-able string (i.e., 'range(10)') or a 
                set of values (i.e., 'A' 'B' 'C' 'D')"""
        parser.add_argument('iter_vals', type=iter_values, help=h)
    
        h = """file providing extra string replaces, with lines of the form 
                 `tag = ['tag1', 'tag2']`; if the keys match keywords in the 
                 template param file, the file with be updated with
                 the `ith` value for the `ith` task"""
        parser.add_argument('--extra', dest='extras', type=extra_iter_values, help=h)
    
        h = "set the logging output to debug, with lots more info printed"
        parser.add_argument('--debug', help=h, action="store_const", dest="log_level", 
                                const=logging.DEBUG, default=logging.INFO)
                                
        return parser.parse_args()
    
    def _initialize_pool_comm(self):
        """
        Internal function that initializes the `MPI.Intracomm` used by the 
        pool of workers. This will be passed to the task function and used 
        in task computation
        """
        # split the ranks
        self.pool_comm = None
        chain_ranks = []
        color = 0
        worker_count = 0
        for i, ranks in split_ranks(self.size, self.config.nodes):
            chain_ranks.append(ranks[0])
            if self.rank in ranks: color = i+1
            worker_count += len(ranks)
        
        if worker_count != self.size-1:
            args = (worker_count, self.size-1)
            raise RuntimeError("mismatch between worker count (%d) and spawned worker processes (%d)" %args)
        self.pool_comm = self.comm.Split(color, 0)
        
    def run_all(self):
        """
        Run all the tasks
        """
        # read the parameter file
        param_file = open(self.config.param_file, 'r').read()
    
        # define MPI message tags
        tags = enum('READY', 'DONE', 'EXIT', 'START')
        status = MPI.Status()
    
        # crash if we only have one process or one node
        if self.size <= self.config.nodes:
            args = (self.size, self.config.nodes+1, self.config.nodes)
            raise ValueError("only have %d ranks; need at least %d to use the desired %d nodes" %args)    
      
        # make the pool comm
        self._initialize_pool_comm()
    
        # the tasks provided on the command line
        tasks = self.config.iter_vals
        num_tasks = len(tasks)
    
        # master distributes the tasks
        if self.rank == 0:
        
            # initialize
            task_index = 0
            num_workers = self.config.nodes
            closed_workers = 0
        
            # loop until all workers have finished with no more tasks
            logger.info("master starting with %d node workers with %d total tasks" %(num_workers, num_tasks))
            while closed_workers < num_workers:
                data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                source = status.Get_source()
                tag = status.Get_tag()
            
                # worker is ready, so send it a task
                if tag == tags.READY:
                    if task_index < num_tasks:
                        self.comm.send(tasks[task_index], dest=source, tag=tags.START)
                        logger.debug("sending task `%s` to worker %d" %(str(tasks[task_index]), source))
                        task_index += 1
                    else:
                        self.comm.send(None, dest=source, tag=tags.EXIT)
                elif tag == tags.DONE:
                    results = data
                    logger.debug("received result from worker %d" %source)
                elif tag == tags.EXIT:
                    closed_workers += 1
                    logger.debug("worker %d has exited, closed workers = %d" %(source, closed_workers))
    
        # worker processes wait and execute single jobs
        else:
            if self.pool_comm.rank == 0:
                args = (self.rank, MPI.Get_processor_name(), self.pool_comm.size)
                logger.info("pool master rank is %d on %s with %d processes available" %args)
            while True:
                task = -1
                tag = -1
        
                # have the master rank of the pool ask for task and then broadcast
                if self.pool_comm.rank == 0:
                    self.comm.send(None, dest=0, tag=tags.READY)
                    task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                    tag = status.Get_tag()
                task = self.pool_comm.bcast(task)
                tag = self.pool_comm.bcast(tag)
        
                # do the work here
                if tag == tags.START:
                    result = self.run_single_task(task, tasks.index(task), param_file)
                    self.pool_comm.Barrier() # wait for everyone
                    if self.pool_comm.rank == 0:
                        self.comm.send(result, dest=0, tag=tags.DONE) # done this task
                elif tag == tags.EXIT:
                    break

            self.pool_comm.Barrier()
            if self.pool_comm.rank == 0:
                self.comm.send(None, dest=0, tag=tags.EXIT) # exiting
    
        # free and exit
        logger.debug("rank %d process finished" %self.rank)
        self.comm.Barrier()
        if self.rank == 0:
            logger.info("master is finished; terminating")
            self.pool_comm.Free()
            
            
    def run_single_task(self, task, itask, param_file):
        """
        Run a single job, calling the task function with the parameters
        specified for this job iteration
    
        Parameters
        ----------
        task :  
            the value of the counter that specifies this iteration
        itask : int
            the integer index of this task
        param_file : str
            the parameter file for the task function, read as a string, 
            which will be string formatted
        """
        pool_rank = self.pool_comm.rank

        # if you are the pool's root, write out the temporary parameter file
        temp_name = None
        if pool_rank == 0:
            # generate the parameter file
            kwargs = [kw for _, kw, _, _ in param_file._formatter_parser() if kw]
            with tempfile.NamedTemporaryFile(delete=False) as ff:
                temp_name = ff.name
                possible_kwargs = {self.config.iter_str : task}
                if self.config.extras is not None:
                    for k in self.config.extras:
                        possible_kwargs[k] = self.config.extras[k][itask]
                valid = {k:v for k,v in possible_kwargs.iteritems() if k in kwargs}
                ff.write(param_file.format(**valid))
        
        # bcast the file name to all in the worker pool
        temp_name = self.pool_comm.bcast(temp_name, root=0)

        # parse the file with updated parameters
        ns = self.task_parser.parse_args(['@%s' %temp_name])

        # run the task function using the comm for this worker pool
        self.task_function(ns, comm=self.pool_comm)

        # remove temporary files
        if pool_rank == 0:
            if os.path.exists(temp_name): 
                os.remove(temp_name)
    
        
        
        
        
