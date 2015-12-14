import logging
import os
import tempfile
from mpi4py import MPI

logger = logging.getLogger('taskmanager')

#------------------------------------------------------------------------------
# tools
#------------------------------------------------------------------------------        
def split_ranks(N_ranks, avg):
    """
    Divide the ranks into chunks, trying to specify `avg` ranks per chunk,
    plus any remainder. This first removes the master (0) rank
    
    Parameters
    ----------
    N_ranks : int
        the total number of ranks available
    avg : int
        the desired number of ranks per chunk
    """
    seq = range(1, N_ranks)
    N = len(seq)

    start = 0
    end = avg
    i = 0
    while start < N:
        if end > N: end = N
        yield i, seq[start:end]
        start = end
        end += avg
        i += 1
        
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
        
    
    @staticmethod
    def replacements_from_file(value):
        """
        Provided an existing file name, read the file into 
        a dictionary. The keys are interpreted as the string format name, 
        and the values are a list of values to iterate over for each job
        """
        if not os.path.exists(value):
            raise RuntimeError("for `replacements_from_file`, file `%s` does not exist" %value)
        toret = {}
        execfile(value, globals(), toret)
        return toret
        
    @staticmethod
    def tasks_parser(value):
        """
        Given a string of the format ``key tasks``, split the string and then
        try to parse the ``tasks``, by first trying to evaluate it, and then
        simply splitting it and interpreting the results as the tasks. 
        
        The general use cases are: 
        
        1) "box range(2)" -> key = `box`, tasks = `[0, 1]`
        2) "box A B C" -> key = `box`, tasks = `['A', 'B', 'C']`
        """
        import yaml
        
        try:
            fields = yaml.load(value)
            keys = list(fields.keys())
            if len(fields) != 1:
                raise Exception
        except:
            raise ValueError("specify iteration tasks via the format: ``-i key: [task1, task2]``")
        
        key = keys[0]
        if isinstance(fields[key], list):
            parsed = fields[key]
        else:
            # try to eval into a list
            try:
                parsed = eval(fields[key])
                if not isinstance(parser, list):
                    raise ValueError("result of `eval` on iteration string should be list" %(fields[key]))
            except:
                raise ValueError("tried but failed to `eval` iteration string `%s`" %(fields[key]))
    
        return [key, parsed]
        
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
        import itertools
        
        # parse
        parser = argparse.ArgumentParser(description=desc) 
          
        h = """the name of the template file holding parameters needed to run the 
                desired task; the file should use ``string.format`` syntax to 
                indicate which variables will be updated for each task, i.e., the
                output file could be specified in the file as 
                'output/pkmu_output_box{box}.dat'"""
        parser.add_argument('param_file', type=str, help=h)
    
        h = "the desired number of workers for each task force"
        parser.add_argument('workers_per', type=int, help=h)
    
        h =  """given a string of the format ``key tasks``, split the string and then
                try to parse the ``tasks``, by first trying to evaluate it, and then
                simply splitting it and interpreting the results as the tasks. 
        
                The general use cases are: 
        
                1) "-i box: range(2)" -> key = `box`, tasks = `[0, 1]`
                2) "-i box: [A, B, C,]" -> key = `box`, tasks = `['A', 'B', 'C']`
                
                If multiple options passed with `-i` flag, then the total tasks to 
                perform will be the product of the tasks lists passed"""
        parser.add_argument('-i', dest='tasks', action='append', type=cls.tasks_parser, required=True, help=h)
    
        h = """file providing extra string replaces, with lines of the form 
                 `tag = ['tag1', 'tag2']`; if the keys match keywords in the 
                 template param file, the file with be updated with
                 the `ith` value for the `ith` task"""
        parser.add_argument('--extra', dest='extras', type=cls.replacements_from_file, help=h)
    
        h = "set the logging output to debug, with lots more info printed"
        parser.add_argument('--debug', help=h, action="store_const", dest="log_level", 
                                const=logging.DEBUG, default=logging.INFO)
                                
        args = parser.parse_args()
        
        # format the tasks, taking the product of multiple task lists
        keys = []; values = []
        for [key, tasks] in args.tasks:
            keys.append(key)
            values.append(tasks)

        # take the product
        if len(keys) > 1:
            values = list(itertools.product(*values))
        else:
            values = values[0]
            
        # save
        args.task_keys = keys
        args.tasks = values
        
        return args    
    
    def _initialize_worker_comm(self):
        """
        Internal function that initializes the `MPI.Intracomm` used by each
        group of workers. This will be passed to the task function and used 
        in task computation
        """
        self.worker_comm = None
        self.num_groups = 0
        
        nranks_min = int(0.5*self.config.workers_per)
        color = 0
        rank_count = 0
        for i, ranks in split_ranks(self.size, self.config.workers_per):
            
            if self.rank in ranks:
                
                # only used this group of workers if it has at least 1/2 the 
                # desired amount
                if len(ranks) >= nranks_min:
                    color = i+1
                else:
                    color = None
            
            rank_count += len(ranks)
            if len(ranks) >= nranks_min:
                self.num_groups += 1
        
        if rank_count != self.size-1:
            args = (rank_count, self.size-1)
            raise RuntimeError("mismatch between rank count (%d) and spawned worker processes (%d)" %args)
        if color is not None:
            self.worker_comm = self.comm.Split(color, 0)
        
    def run_all(self):
        """
        Run all the tasks
        """
        # read the parameter file
        param_file = open(self.config.param_file, 'r').read()
    
        # define MPI message tags
        tags = enum('READY', 'DONE', 'EXIT', 'START')
        status = MPI.Status()
    
        # crash if we don't have enough cpus
        if self.size <= self.config.workers_per:
            args = (self.size, self.config.workers_per+1)
            raise ValueError("only have %d ranks; need at least %d" %args)    
      
        # make the pool comm
        self._initialize_worker_comm()
    
        # the tasks provided on the command line
        tasks = self.config.tasks
        num_tasks = len(tasks)
    
        # master distributes the tasks
        if self.rank == 0:
        
            # initialize
            task_index = 0
            num_groups = self.num_groups
            closed_groups = 0
        
            # loop until all workers have finished with no more tasks
            logger.info("master starting with %d worker groups with %d total tasks" %(num_groups, num_tasks))
            while closed_groups < num_groups:
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
                    closed_groups += 1
                    logger.debug("worker %d has exited, closed workers = %d" %(source, closed_groups))
    
        # worker processes wait and execute single jobs
        # but leftover processes dont do anything here
        elif self.worker_comm is not None:
            
            if self.worker_comm.rank == 0:
                args = (self.rank, MPI.Get_processor_name(), self.worker_comm.size)
                logger.info("pool master rank is %d on %s with %d processes available" %args)
            while True:
                task = -1
                tag = -1
        
                # have the master rank of the pool ask for task and then broadcast
                if self.worker_comm.rank == 0:
                    self.comm.send(None, dest=0, tag=tags.READY)
                    task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                    tag = status.Get_tag()
                task = self.worker_comm.bcast(task)
                tag = self.worker_comm.bcast(tag)
        
                # do the work here
                if tag == tags.START:
                    result = self.run_single_task(task, tasks.index(task), param_file)
                    self.worker_comm.Barrier() # wait for everyone
                    if self.worker_comm.rank == 0:
                        self.comm.send(result, dest=0, tag=tags.DONE) # done this task
                elif tag == tags.EXIT:
                    break

            self.worker_comm.Barrier()
            if self.worker_comm.rank == 0:
                self.comm.send(None, dest=0, tag=tags.EXIT) # exiting
    
        # free and exit
        logger.debug("rank %d process finished" %self.rank)
        self.comm.Barrier()
        if self.rank == 0:
            logger.info("master is finished; terminating")
            self.worker_comm.Free()
            
            
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
        pool_rank = self.worker_comm.rank

        # if you are the pool's root, write out the temporary parameter file
        temp_name = None
        if pool_rank == 0:
            # extract the keywords that we need to format from template file
            kwargs = [kw for _, kw, _, _ in param_file._formatter_parser() if kw]
            
            # initialize a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as ff:
                temp_name = ff.name
                
                # key/values for this task 
                if len(self.config.task_keys) == 1:
                    possible_kwargs = {self.config.task_keys[0] : task}
                else:
                    possible_kwargs = dict(zip(self.config.task_keys, task))
                    
                # any extra key/value pairs for this tasks
                if self.config.extras is not None:
                    for k in self.config.extras:
                        possible_kwargs[k] = self.config.extras[k][itask]
                        
                # do the string formatting if the key is present in template
                valid = {k:v for k,v in possible_kwargs.iteritems() if k in kwargs}
                ff.write(param_file.format(**valid))
        
        # bcast the file name to all in the worker pool
        temp_name = self.worker_comm.bcast(temp_name, root=0)

        # parse the file with updated parameters
        ns = self.task_parser.parse_args(['%s' %temp_name])

        # run the task function using the comm for this worker pool
        self.task_function(ns, comm=self.worker_comm)

        # remove temporary files
        if pool_rank == 0:
            if os.path.exists(temp_name): 
                os.remove(temp_name)
    
        
        
        
        
