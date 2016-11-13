#! /usr/bin/env python

import logging
import os
import tempfile
from string import Formatter

from mpi4py import MPI
from nbodykit.core import Algorithm
from nbodykit import algorithms
from nbodykit.utils.taskmanager import TaskManager
from nbodykit.plugins.fromfile import ReadConfigFile

# setup the logging

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

logger = logging.getLogger('nbkit-batch')

def replacements_from_file(value):
    """
    Provided an existing file name, read the file into 
    a dictionary. The keys are interpreted as the string format name, 
    and the values are a list of values to iterate over for each job
    """
    if not os.path.exists(value):
        raise RuntimeError("for `replacements_from_file`, file `%s` does not exist" %value)
    
    toret = {}
    with open(value) as f:
        code = compile(f.read(), value, 'exec')
        exec(code, globals(), toret)

    return toret
        
def tasks_parser(value):
    """
    Given a string of the format ``key tasks``, split the string and then
    try to parse the ``tasks``, by first trying to evaluate it, and then
    simply splitting it and interpreting the results as the tasks. 
    
    The general use cases are: 
    
    1) "box: range(2)" -> key = `box`, tasks = `[0, 1]`
    2) "box: ['A', 'B' 'C']" -> key = `box`, tasks = `['A', 'B', 'C']`
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
            if not isinstance(parsed, list):
                raise ValueError("result of `eval` on iteration string should be list" %(fields[key]))
        except:
            raise ValueError("tried but failed to `eval` iteration string `%s`" %(fields[key]))

    return [key, parsed]

def SafeStringParse(formatter, s, keys):
    """
    A "safe" version :func:`string.Formatter.parse` that will 
    only parse the input keys specified in ``keys``
    
    Parameters
    ----------
    formatter : string.Formatter
        the string formatter class instance
    s : str
        the string we are formatting
    keys : list of str
        list of the keys to accept as valid
    """
    # the default list of keys
    l = list(Formatter.parse(formatter, s))
    
    toret = []
    for x in l:
        if x[1] in keys:
            toret.append(x)
        else:
            val = x[0]
            if x[1] is not None:
                fmt = "" if not x[2] else ":%s" %x[2]
                val += "{%s%s}" %(x[1], fmt)
            toret.append((val, None, None, None))
    return iter(toret)

class BatchAlgorithmDriver(object):
    """
    Class to facilitate running algorithms in batch mode
    """
    def __init__(self, comm, 
                       algorithm_name, 
                       config, 
                       cpus_per_worker, 
                       task_dims, 
                       task_values, 
                       log_level=logging.INFO, 
                       extras={},
                       use_all_cpus=False):
        """
        Parameters
        ----------
        comm : MPI communicator
            the global communicator that will be split and divided
            amongs the independent workers
        algorithm_name : str
            the string name of the `Algorithm` we are running
        config : str
            the name of the file holding the template config file, which
            will be updated for each task that is performed
        cpus_per_worker : int
            the desired number of ranks assigned to each independent
            worker, when iterating over the tasks in parallel
        task_dims : list
            a list of strings specifying the names of the task dimensions -- 
            these specify the string formatting key when updating the config
            template file for each task value
        task_value : list
            a list of tuples specifying the task values which will be iterated 
            over -- each tuple should be the length of `task_dims`
        log_level : int, optional
            an integer specifying the logging level to use -- default
            is the `INFO` level
        extras : dict, optional
            a dictionary where the values are lists of string replacements, with
            length equal to the total number of tasks -- if the keys are present
            in the config file, the string formatting will update the config
            file with the `ith` element of the list for the `ith` iteration
        use_all_cpus : bool, optional
            if `True`, then do not force each worker group to have 
            exactly `cpus_per_worker` ranks, instead including the remainder 
            as well; default is `False`
        """
        setup_logging(log_level)

        self.algorithm_name  = algorithm_name
        self.algorithm_class = getattr(algorithms, algorithm_name)
        self.template        = os.path.expandvars(open(config, 'r').read())
        self.cpus_per_worker = cpus_per_worker
        self.task_dims       = task_dims
        self.task_values     = task_values
        self.extras          = extras
        
        self.comm      = comm
        self.size      = comm.size
        self.rank      = comm.rank
        
        # initialize the worker pool
        kws = {'comm':self.comm, 'use_all_cpus':use_all_cpus}
        if log_level <= logging.DEBUG: kws['debug'] = True
        self.workers = TaskManager(self.compute_one_task, self.cpus_per_worker, **kws)
                
    @classmethod
    def create(cls, comm=None, desc=None):
        """
        Parse the task manager and return the ``BatchAlgorithmDriver`` instance
        """
        import inspect 
        
        if comm is None: comm = MPI.COMM_WORLD
        args_dict = cls.parse_args(desc)
        args_dict['comm'] = comm
        
        # inspect the __init__ function
        args, varargs, varkw, defaults = inspect.getargspec(cls.__init__)
        
        # determine the required arguments
        args = args[1:] # remove 'self'
        if defaults:
            required = args[:-len(defaults)]
        else:
            required = args
            
        # get the args, kwargs to pass to __init__
        fargs = tuple(args_dict[p] for p in required)
        fkwargs = {}
        if defaults:
            for i, p in enumerate(defaults):
                name = args[-len(defaults)+i]
                fkwargs[name] = args_dict.get(name, defaults[i])
        
        return cls(*fargs, **fkwargs)
        
    @classmethod
    def parse_args(cls, desc=None):
        """
        Parse command-line arguments that are needed to initialize a 
        `BatchAlgorithmDriver` class
        
        Parameters
        ----------
        desc : str, optional
            the description of to use for this parser
        """
        import argparse
        import itertools
        
        # parse
        parser = argparse.ArgumentParser(description=desc) 
        
        # first argument is the algorithm name
        h = 'the name of the `Algorithm` to run in batch mode'
        valid_algorithms = list(vars(algorithms))
        parser.add_argument(dest='algorithm_name', choices=valid_algorithms, help=h)  
        
        # the number of independent workers
        h = """the desired number of ranks assigned to each independent
                worker, when iterating over the tasks in parallel""" 
        parser.add_argument('cpus_per_worker', type=int, help=h)
    
        # now do the required named arguments
        required_named = parser.add_argument_group('required named arguments')
        
        # specify the tasks along one dimension 
        h =  """given a string of the format ``key: tasks``, split the string and then
                try to parse the ``tasks``, by first trying to evaluate it, and then
                simply splitting it and interpreting the results as the tasks. 
        
                The general use cases are: 
        
                1) "box: range(2)" -> key = `box`, tasks = `[0, 1]`
                2) "box: [A, B, C]" -> key = `box`, tasks = `['A', 'B', 'C']`
                
                If multiple options passed with `-i` flag, then the total tasks to 
                perform will be the product of the tasks lists passed"""
        required_named.add_argument('-i', dest='tasks', action='append', 
                type=tasks_parser, required=True, help=h)
    
        # the template config file
        h = """the name of the template config file (using YAML synatx) that 
                provides the `Algorithm` parameters; the file should use 
                ``string.format`` syntax to indicate which variables will be 
                updated for each task, i.e., an input file could be specified 
                as 'input/DataSource_box{box}.dat', if `box` were one of the task 
                dimensions"""
        required_named.add_argument('-c', '--config', required=True, type=str, help=h)
    
        # read any extra string replacements from file
        h = """file providing extra string replaces, with lines of the form 
                 `tag = ['tag1', 'tag2']`; if the keys match keywords in the 
                 template param file, the file with be updated with
                 the `ith` value for the `ith` task"""
        parser.add_argument('--extras', dest='extras', default={}, type=replacements_from_file, help=h)
    
        h = "set the logging output to debug, with lots more info printed"
        parser.add_argument('--debug', help=h, action="store_const", dest="log_level", 
                            const=logging.DEBUG, default=logging.INFO)
                            
        h = "if `True`, include all available cpus in the worker pool"
        parser.add_argument('--use_all_cpus', action='store_true', help=h)
                                
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
        args.task_dims = keys
        args.task_values = values
        
        return vars(args)
            
    def compute(self):
        """
        Compute all of the tasks
        """             
        # compute the work
        results = self.workers.compute(self.task_values)
        return results
            
    def compute_one_task(self, itask, task):
        """
        Run the algorithm once, using the parameters specified by `task`,
        which is the `itask` iteration
    
        Parameters
        ----------
        itask : int
            the integer index of this task
        task : tuple
            a tuple of values representing this task value
        """
        # if you are the pool's root, write out the temporary parameter file
        if self.workers.subcomm.rank == 0:
                
            # key/values for this task 
            if len(self.task_dims) == 1:
                possible_kwargs = {self.task_dims[0] : task}
            else:
                possible_kwargs = dict(zip(self.task_dims, task))
                
            # any extra key/value pairs for this tasks
            if self.extras is not None:
                for k in self.extras:
                    possible_kwargs[k] = self.extras[k][itask]
                    
            # use custom formatter that only formats the possible keys, ignoring other
            # occurences of curly brackets
            formatter = Formatter()
            formatter.parse = lambda l: SafeStringParse(formatter, l, list(possible_kwargs))
            kwargs = [kw for _, kw, _, _ in formatter.parse(self.template) if kw]
                    
            # do the string formatting if the key is present in template
            valid = {k:possible_kwargs[k] for k in possible_kwargs if k in kwargs}
            config_stream = formatter.format(self.template, **valid)
        else:
            config_stream = None

        # bcast the file stream to all in the worker pool
        config_stream = self.workers.subcomm.bcast(config_stream, root=0)

        # configuration file passed via -c
        params, extra = ReadConfigFile(config_stream, self.algorithm_class.schema)
        
        # output is required
        output = getattr(extra, 'output', None)
        if output is None:
            raise ValueError("argument `output` is required in config file")
            
        # initialize the algorithm and run
        alg = self.algorithm_class(**vars(params))
        result = alg.run()
        alg.save(output, result)
                
        return 0

   
if __name__ == '__main__' :
    
    desc = """iterate (possibly in parallel) over a set of configuration parameters, 
              running the specified `Algorithm` for each"""
    batch = BatchAlgorithmDriver.create(MPI.COMM_WORLD, desc=desc)
    batch.compute()
    
    
