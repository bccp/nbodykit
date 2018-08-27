import os
import traceback
import logging
import numpy
from mpi4py import MPI
from nbodykit import CurrentMPIComm

def split_ranks(N_ranks, N, include_all=False):
    """
    Divide the ranks into chunks, attempting to have `N` ranks
    in each chunk. This removes the master (0) rank, such
    that `N_ranks - 1` ranks are available to be grouped

    Parameters
    ----------
    N_ranks : int
        the total number of ranks available
    N : int
        the desired number of ranks per worker
    include_all : bool, optional
        if `True`, then do not force each group to have
        exactly `N` ranks, instead including the remainder as well;
        default is `False`
    """
    available = list(range(1, N_ranks)) # available ranks to do work
    total = len(available)
    extra_ranks = total % N

    if include_all:
        for i, chunk in enumerate(numpy.array_split(available, max(total//N, 1))):
            yield i, list(chunk)
    else:
        for i in range(total//N):
            yield i, available[i*N:(i+1)*N]

        i = total // N
        if extra_ranks and extra_ranks >= N//2:
            remove = extra_ranks % 2 # make it an even number
            ranks = available[-extra_ranks:]
            if remove: ranks = ranks[:-remove]
            if len(ranks):
                yield i+1, ranks

def enum(*sequential, **named):
    """
    Enumeration values to serve as status tags passed
    between processeseee
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


class TaskManager(object):
    """
    An MPI task manager that distributes tasks over a set of MPI processes,
    using a specified number of independent workers to compute each task.

    Given the specified number of independent workers (which compute
    tasks in parallel), the total number of available CPUs will be
    divided evenly.

    The main function is ``iterate`` which iterates through a set of tasks,
    distributing the tasks in parallel over the available ranks.

    Parameters
    ----------
    cpus_per_task : int
        the desired number of ranks assigned to compute
        each task
    comm : MPI communicator, optional
        the global communicator that will be split so each worker
        has a subset of CPUs available; default is COMM_WORLD
    debug : bool, optional
        if `True`, set the logging level to `DEBUG`, which prints
        out much more information; default is `False`
    use_all_cpus : bool, optional
        if `True`, use all available CPUs, including the remainder
        if `cpus_per_task` is not divide the total number of CPUs
        evenly; default is `False`
    """
    logger = logging.getLogger('TaskManager')

    @CurrentMPIComm.enable
    def __init__(self, cpus_per_task, comm=None, debug=False, use_all_cpus=False):

        if debug:
            self.logger.setLevel(logging.DEBUG)

        self.cpus_per_task = cpus_per_task
        self.use_all_cpus  = use_all_cpus

        # the base communicator
        self.basecomm = MPI.COMM_WORLD if comm is None else comm
        self.rank     = self.basecomm.rank
        self.size     = self.basecomm.size

        # need at least one
        if self.size == 1:
            raise ValueError("need at least two processes to use a TaskManager")

        # communication tags
        self.tags = enum('READY', 'DONE', 'EXIT', 'START')

        # the task communicator
        self.comm = None

        # store a MPI status
        self.status = MPI.Status()

    def __enter__(self):
        """
        Split the base communicator such that each task gets allocated
        the specified number of cpus to perform the task with
        """
        chain_ranks = []
        color = 0
        total_ranks = 0
        nworkers = 0

        # split the ranks
        for i, ranks in split_ranks(self.size, self.cpus_per_task, include_all=self.use_all_cpus):
            chain_ranks.append(ranks[0])
            if self.rank in ranks: color = i+1
            total_ranks += len(ranks)
            nworkers = nworkers + 1
        self.workers = nworkers # store the total number of workers

        # check for no workers!
        if self.workers == 0:
            raise ValueError("no pool workers available; try setting `use_all_cpus` = True")

        leftover = (self.size - 1) - total_ranks
        if leftover and self.rank == 0:
            args = (self.cpus_per_task, self.size-1, leftover)
            self.logger.warning("with `cpus_per_task` = %d and %d available rank(s), %d rank(s) will do no work" %args)
            self.logger.warning("set `use_all_cpus=True` to use all available cpus")

        # crash if we only have one process or one worker
        if self.size <= self.workers:
            args = (self.size, self.workers+1, self.workers)
            raise ValueError("only have %d ranks; need at least %d to use the desired %d workers" %args)

        # ranks that will do work have a nonzero color now
        self._valid_worker = color > 0

        # split the comm between the workers
        self.comm = self.basecomm.Split(color, 0)
        CurrentMPIComm.push(self.comm)

        return self

    def is_root(self):
        """
        Is the current process the root process?

        Root is responsible for distributing the tasks to the other available ranks
        """
        return self.rank == 0

    def is_worker(self):
        """
        Is the current process a valid worker?

        Workers wait for instructions from the master
        """
        try:
            return self._valid_worker
        except:
            raise ValeuError("workers are only defined when inside the ``with TaskManager()`` context")

    def _get_tasks(self):
        """
        Internal generator that yields the next available task from a worker
        """
        if self.is_root():
            raise RuntimeError("Root rank mistakenly told to await tasks")

        # logging info
        if self.comm.rank == 0:
            args = (self.rank, MPI.Get_processor_name(), self.comm.size)
            self.logger.debug("worker master rank is %d on %s with %d processes available" %args)

        # continously loop and wait for instructions
        while True:
            args = None
            tag = -1

            # have the master rank of the subcomm ask for task and then broadcast
            if self.comm.rank == 0:
                self.basecomm.send(None, dest=0, tag=self.tags.READY)
                args = self.basecomm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
                tag = self.status.Get_tag()

            # bcast to everyone in the worker subcomm
            args  = self.comm.bcast(args) # args is [task_number, task_value]
            tag   = self.comm.bcast(tag)

            # yield the task
            if tag == self.tags.START:

                # yield the task value
                yield args

                # wait for everyone in task group before telling master this task is done
                self.comm.Barrier()
                if self.comm.rank == 0:
                    self.basecomm.send([args[0], None], dest=0, tag=self.tags.DONE)

            # see ya later
            elif tag == self.tags.EXIT:
                break

        # wait for everyone in task group and exit
        self.comm.Barrier()
        if self.comm.rank == 0:
            self.basecomm.send(None, dest=0, tag=self.tags.EXIT)

        # debug logging
        self.logger.debug("rank %d process is done waiting" %self.rank)

    def _distribute_tasks(self, tasks):
        """
        Internal function that distributes the tasks from the root to the workers
        """
        if not self.is_root():
            raise ValueError("only the root rank should distribute the tasks")

        ntasks = len(tasks)
        task_index     = 0
        closed_workers = 0

        # logging info
        args = (self.workers, ntasks)
        self.logger.debug("master starting with %d worker(s) with %d total tasks" %args)

        # loop until all workers have finished with no more tasks
        while closed_workers < self.workers:

            # look for tags from the workers
            data = self.basecomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
            source = self.status.Get_source()
            tag = self.status.Get_tag()

            # worker is ready, so send it a task
            if tag == self.tags.READY:

                # still more tasks to compute
                if task_index < ntasks:
                    this_task = [task_index, tasks[task_index]]
                    self.basecomm.send(this_task, dest=source, tag=self.tags.START)
                    self.logger.debug("sending task `%s` to worker %d" %(str(tasks[task_index]), source))
                    task_index += 1

                # all tasks sent -- tell worker to exit
                else:
                    self.basecomm.send(None, dest=source, tag=self.tags.EXIT)

            # store the results from finished tasks
            elif tag == self.tags.DONE:
                self.logger.debug("received result from worker %d" %source)

            # track workers that exited
            elif tag == self.tags.EXIT:
                closed_workers += 1
                self.logger.debug("worker %d has exited, closed workers = %d" %(source, closed_workers))

    def iterate(self, tasks):
        """
        A generator that iterates through a series of tasks in parallel.

        Notes
        -----
        This is a collective operation and should be called by
        all ranks

        Parameters
        ----------
        tasks : iterable
            an iterable of `task` items that will be yielded in parallel
            across all ranks

        Yields
        -------
        task :
            the individual items of `tasks`, iterated through in parallel
        """
        # master distributes the tasks and tracks closed workers
        if self.is_root():
            self._distribute_tasks(tasks)

        # workers will wait for instructions
        elif self.is_worker():
            for tasknum, args in self._get_tasks():
                yield args

    def map(self, function, tasks):
        """
        Like the built-in :func:`map` function, apply a function to all
        of the values in a list and return the list of results.

        If ``tasks`` contains tuples, the arguments are passed to
        ``function`` using the ``*args`` syntax

        Notes
        -----
        This is a collective operation and should be called by
        all ranks

        Parameters
        ----------
        function : callable
            The function to apply to the list.
        tasks : list
            The list of tasks

        Returns
        -------
        results : list
            the list of the return values of :func:`function`
        """
        results = []

        # master distributes the tasks and tracks closed workers
        if self.is_root():
            self._distribute_tasks(tasks)

        # workers will wait for instructions
        elif self.is_worker():

            # iterate through tasks in parallel
            for tasknum, args in self._get_tasks():

                # make function arguments consistent with *args
                if not isinstance(args, tuple):
                    args = (args,)

                # compute the result (only worker root needs to save)
                result = function(*args)
                if self.comm.rank == 0:
                    results.append((tasknum, result))

        # put the results in the correct order
        results = self.basecomm.allgather(results)
        results = [item for sublist in results for item in sublist]
        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exit gracefully by closing and freeing the MPI-related variables
        """
        if exc_value is not None:
            trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback, limit=5))
            self.logger.error("an exception has occurred on rank %d:\n%s" %(self.rank, trace))

            # bit of hack that forces mpi4py to exit all ranks
            # see https://groups.google.com/forum/embed/#!topic/mpi4py/RovYzJ8qkbc
            os._exit(1)

        # wait and exit
        self.logger.debug("rank %d process finished" %self.rank)
        self.basecomm.Barrier()

        if self.is_root():
            self.logger.debug("master is finished; terminating")

        CurrentMPIComm.pop()

        if self.comm is not None:
            self.comm.Free()
