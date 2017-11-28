.. _parallel-computation:

Parallel Computation with nbodykit
==================================

nbodykit is fully parallelized and is built on top of the Message Passage Interface
(MPI) using the Python bindings provided by the :mod:`mpi4py` library.
In this section, we discuss some of the crucial ways in which nbodykit
interacts with MPI. For those unfamiliar with MPI, a good place to start is the
`documentation for mpi4py <http://mpi4py.readthedocs.io/en/stable/intro.html>`_.

Running nbodykit in parallel
----------------------------

Users can execute Python scripts using nbodykit in parallel
using the standardized ``mpiexec`` MPI executable. If installing nbodykit
through Anaconda (as described :ref:`here <conda-installation>`), the
``mpiexec`` command will be available for use in your conda environment.

As an example, we can run
`this example script <https://raw.githubusercontent.com/bccp/nbodykit/master/nersc/example.py>`_
in parallel. The example script can be downloaded manually from GitHub, or
using the ``wget`` utility:

.. code:: bash

    # install wget via conda, if not installed
    $ conda install wget

    # download the example script
    $ wget https://raw.githubusercontent.com/bccp/nbodykit/master/nersc/example.py

We can execute the example script with 4 MPI cores in parallel using:

.. code:: bash

    # run with 4 MPI workers
    $ mpiexec -n 4 python example.py

This example script generates a set of simulated objects and computes the
power spectrum via the :class:`~nbodykit.algorithms.fftpower.FFTPower`
algorithm, saving the result to a JSON file ``nbkit_example_power.json``.

The MPI calling sequence typically differs for supercomputing environments
and depends on the task scheduler being used. For example, when using the
`Slurm manager <https://slurm.schedmd.com>`_ (as on NERSC), the equivalent
command to execute the example script is:

.. code:: bash

    $ srun -n 4 python example.py

On a managed computing facility, it is also usually necessary to include
directives that reserves the computing resource used for running the script.
Often, you can refer to the 'Submitting a Job' section of the user guide of
the facility, e.g., `NERSC Cori`_, `NERSC Edison`_, and the `NCSA`_.
However, be aware these user guides are not always accurate,
and it is always better to check with someone who uses these facilities first.


.. _NERSC Cori: http://www.nersc.gov/users/computational-systems/cori/running-jobs/
.. _NERSC Edison: http://www.nersc.gov/users/computational-systems/edison/running-jobs/
.. _NCSA: https://bluewaters.ncsa.illinois.edu/getting-started/#Running

A Primer on MPI Communication
-----------------------------

MPI stands for Message Passage Interface, and unsurprisingly, one of its
key elements is the communication between processes running in parallel.
The MPI standard allows processes running in parallel, which own their
own memory, to exchange messages, thus allowing the independent results
computed by individual processes to be combined into a single result.

The MPI communicator object is responsible for managing the communication
of data and messages between parallel processes. In nbodykit, we manage the
current MPI communicator using the :class:`nbodykit.CurrentMPIComm` class.
By default, all nbodykit objects, i.e., catalog and mesh objects, use
the communicator defined by this class to exchange results. For
catalog and mesh objects, the communicator is stored as the :attr:`comm`
attribute.

Users can access the current communicator by calling the
:func:`~nbodykit.CurrentMPIComm.get` function of the
:class:`~nbodykit.CurrentMPIComm` object. For example,

.. code:: python

    from nbodykit import CurrentMPIComm
    comm = CurrentMPIComm.get()

The communicator object carries a ``rank`` attribute, which provides
a unique numbering of the available processes controlled
by the communicator, and a ``size`` attribute giving the total number of
processes within the communicator. Often, the ``rank`` attribute
is used to reduce the amount of messages printed to the terminal, e.g.

.. code-block:: python

    if comm.rank == 0:
        print("I am Groot.")

In this case, we get only one print statement, whereas we would get as many as
``comm.size`` messages without the condition. The
`tutorials <http://mpi4py.readthedocs.io/en/stable/tutorial.html>`_
provided in the :mod:`mpi4py` documentation provide
more examples illustrating the power of MPI communicators.

For more advanced users, the current communicator can be set using
the :func:`~nbodykit.CurrentMPIComm.set` function of the
:class:`~nbodykit.CurrentMPIComm` object. This can be useful if the
default communicator (which includes all processes) is split into
sub-communicators. This framework is how we implement the task-based
parallelism provided by the :class:`~nbodykit.batch.TaskManager` object
(see the :ref:`task-based-parallelism` section below). Setting the current
MPI communicator in this manner only affects the creation of objects
after the comm has been set.

.. _data-based-parallelism:

Data-based parallelism
----------------------

When writing code with nbodykit, it's important to keep in mind that memory is
not shared across different CPUs when using MPI. This is particularly relevant
when interacting with data in nbodykit. Both the catalog and mesh objects
are *distributed* containers, meaning that the data is spread out evenly
across the available processes within an MPI communicator. A single
process does not have access to the full catalog or 3D mesh but merely the
portion stored in its memory. A crucial benefit of the distributed nature of
data in nbodykit is that we can quickly load large data sets that would
otherwise not fit into the memory of a single process.

When working with several processes, a simple way to gather the full data set
onto every process is to combine the :func:`numpy.concatenate` function
with the :func:`allgather` method of the current MPI communicator.
For example, given a catalog object, we can gather the full "Position" column
to all ranks using

.. code-block:: python

    data = numpy.concatenate(catalog.comm.allgather(catalog['Position'].compute()), axis=0)

.. important::

  Beware of such :func:`allgather` operations. Each process gets a full copy
  of the data, and the computer will quickly run out of memory if the catalog is large.

.. _task-based-parallelism:

Task-based parallelism
----------------------

Often, large-scale structure data analysis involves hundreds to thousands
of iterations of a single, less computationally expensive task.
Examples include parameter sampling and minimization and the calculation of
clustering statistics from thousands of simulations to determine
covariance properties. nbodykit includes the
:class:`~nbodykit.batch.TaskManager` class to allow
users to easily iterate over multiple tasks while using nbodykit.
Users can specify the desired number of MPI ranks per task, and tasks will
run in parallel ensuring that all MPI ranks are being used.

We attempt to hide most of the MPI complexities from the user by
implementing the :class:`~nbodykit.batch.TaskManager` utility as a Python
context manager. This allows users to simply paste the workflow of a single
task into the context of a :class:`~nbodykit.batch.TaskManager` to iterate
through tasks in parallel.

For example, we can compute the power spectrum of a simulated catalog of
particles with several different bias values using:

.. code-block:: python

  from nbodykit.lab import *

  # the bias values to iterate over
  biases = [1.0, 2.0, 3.0, 4.0]

  # initialize the task manager to run the tasks
  with TaskManager(cpus_per_task=2, use_all_cpus=True) as tm:

    # set up the linear power spectrum
    redshift = 0.55
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')

    # iterate through the bias values
    for bias in tm.iterate(biases):

      # initialize the catalog for this bias
      cat = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=1380., Nmesh=256, bias=bias)

      # compute the power spectrum
      r = FFTPower(cat, mode="2d")

      # and save
      r.save("power-" + str(bias) + ".json")

Here, we rely on the :func:`~nbodykit.batch.TaskManager.iterate` function to
iterate through our tasks using a for loop. We could have also used the
:func:`~nbodykit.batch.TaskManager.map` function to apply a function to each
bias value (it behaves identical to the built-in :func:`map`).

The :class:`~nbodykit.batch.TaskManager` utility works by first splitting
the available CPUs in to subgroups, where the size of the subgroups is determined
by the ``cpus_per_task`` task argument passed to :class:`~nbodykit.batch.TaskManager`.
With these subgroups ready to execute the tasks, the manager uses one root process
to distribute the tasks to one of the subgroups of workers as they become available
and stops when all of the tasks have finished. Internally, the
:class:`~nbodykit.batch.TaskManager` class splits the global MPI communicator
into smaller groups, so that each subgroup of processes can communicate only
with themselves.

.. important::

    Users should execute all of the nbodykit-related code from within
    the context of the :class:`~nbodykit.batch.TaskManager`. In particular,
    users should take care to ensure that all of the catalog and mesh objects
    are created from within the manager's context. Otherwise, there will be
    issues with the MPI communication and the code is likely to stall.

The above code is available in the nbodykit source code on GitHub
`here <https://raw.githubusercontent.com/bccp/nbodykit/master/nersc/example-batch.py>`_.
We encourage users to download the script and experiment with different
``cpus_per_task`` values and total MPI processes. For example, if we
run with 3 total processes and use a single process per task, then two tasks
will always be executed simultaneously (one process is reserved to
distribute the tasks to the other ranks). When running with this configuration,
users will see the following output:

::

    $ mpiexec -n 3 python example-batch.py 1
    rank 2: computing for bias = 1.0
    rank 1: computing for bias = 2.0
    rank 2: computing for bias = 3.0
    rank 1: computing for bias = 4.0
