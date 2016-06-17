Running an Algorithm
====================

An nbodykit :class:`~nbodykit.extensionpoints.Algorithm` can be run using the `nbkit.py`_
executable in the ``bin`` directory. The user can ask for help with the calling signature 
of the script in the usual way:

.. code-block:: bash

    python bin/nbkit.py -h
    
The first argument gives the name of the algorithm plugin that the user wishes to execute, 
while the second argument gives the name of the file to read configuration parameters
from (if no file name is given, the script will read from standard input). For a
discussion of the parsing of configuration files, see `Writing configuration files`_.

The ``nbkit.py`` script also provides an interface for getting help on extension points
and individual plugins. A list of the configuration parameters for the built-in plugins of 
each extension point can be accessed by:

.. code-block:: bash

    # prints help for all DataSource plugins
    python bin/nbkit.py --list-datasources 
    
    # prints help for all Algorithm plugins
    python bin/nbkit.py --list-algorithms 
    
    # prints help for all Painter plugins
    python bin/nbkit.py --list-painters
    
    # prints help for all Transfer plugins
    python bin/nbkit.py --list-transfers

and the help message for an individual plugin can be printed by passing the
plugin name to the ``--list-*`` option, i.e., 

.. code-block:: bash
    
    # prints help message for only the FFTPower algorithm
    python bin/nbkit.py --list-algorithms FFTPower
    
will print the help message for the :class:`FFTPower <nbodykit.plugins.algorithms.PeriodicBox.FFTPowerAlgorithm>`
algorithm. Similarly, the help messages for specific algorithms can also be accessed by 
passing the algorithm name and the ``-h`` option:

.. code-block:: bash

    python bin/nbkit.py FFTPower -h


Using MPI
---------

The nbodykit is designed to be run in parallel using the Message Passage Interface (MPI)
and the python package `mpi4py`_. The executable ``nbkit.py`` can take advantage of
multiple processors to run algorithms in parallel. The usage for running with `n` 
processors is:


.. code-block:: bash

    mpirun -n [n] python bin/nbkit.py ...


Writing configuration files
---------------------------

The parameters needed to execute the desired algorithms should be stored in a file
and passed to the ``nbkit.py`` file as the second argument. The configuration file 
should be written using `YAML`_, which relies on the ``name: value``
syntax to parse (key, value) pairs into dictionaries in Python. 

By example
~~~~~~~~~~

The YAML syntax is best learned by example. Let's consider the 
:class:`FFTPower <nbodykit.plugins.algorithms.PeriodicBox.FFTPowerAlgorithm>` algorithm, which 
computes the power spectrum of two data fields using a
Fast Fourier Transform in a periodic box. The necessary parameters to initialize and
run this algorithm can be accessed from the `schema` attribute of the `FFTPower` class:

.. ipython:: python

    # import the NameSpace holding the loaded algorithms
    from nbodykit.extensionpoints import algorithms
    
    # can also use algorithms.FFTPower? in IPython
    print(algorithms.FFTPower.schema)

An example configuration file for this algorithm is given below. The algorithm reads in 
two data files using the 
:class:`FastPM <nbodykit.plugins.datasource.FastPM.FastPMDataSource>` DataSource and
:class:`FOFGroups <nbodykit.plugins.datasource.FOFGroups.FOFDataSource>` DataSource classes 
and computes the cross power spectrum of the density fields.

.. literalinclude:: ../examples/power/test_cross_power.params
    :linenos:
    :name: config-file

The key aspect of YAML syntax for nbodykit configuration files is that 
parameters listed at a common indent level will be parsed together 
into a dictionary. This is illustrated explicitly with the `cosmo` keyword in 
line 4, which could have been equivalently expressed as:

.. code::
    
    cosmo: 
        Om0: 0.27
        H0: 100

A few other things to note:

    * The names of the parameters given in the configuration file must exactly match the names of the attributes listed in the algorithm's `schema`.
    * All required parameters must be listed in the configuration file, otherwise the code will raise an exception. 
    * The `field` and `other` parameters in this example have subfields, named `DataSource`, `Painter`, and `Transfer`. The parameters that are subfields must be indented from the their parent parameters to indicate that they are subfields.
    * Environment variables can be used in configuration files, using the syntax ``${ENV_VAR}`` or ``$ENV_VAR``. In the above file, both `NBKIT_CACHE` and `NBKIT_HOME` are assumed to be environment variables.

Plugin representations
~~~~~~~~~~~~~~~~~~~~~~

A key aspect of the nbodykit code is the use of plugins; representing them properly
in configuration files is an important step in becoming a successful nbodykit user. 

The function responsible for initializing plugins from their configuration file 
representation is :func:`~nbodykit.extensionpoints.PluginMount.from_config`. This function 
accepts several different ways of representing plugins, and we will illustrate these
methods using the previous :ref:`configuration file <config-file>`.

1. The parameters needed to initialize a plugin can be given at a common indent level, and the 
keyword `plugin` can be used to give the name of the plugin to load. This is illustrated
for the `field.DataSource` parameter, which will be loaded into a :class:`FastPM <nbodykit.plugins.datasource.FastPM.FastPMDataSource>` DataSource:

.. literalinclude:: ../examples/power/test_cross_power.params
    :lines: 7-10

    
2. Rather than using the `plugin` parameter to give the name of the plugin to load, the user
can indent the plugin arguments under the name of the plugin, as is illustrated below for
the :class:`FOFGroups <nbodykit.plugins.datasource.FOFGroups.FOFDataSource>` DataSource:

.. literalinclude:: ../examples/power/test_cross_power.params
    :lines: 17-22
    
3.  If the plugin needs no arguments to be intialized, the user can simply use the name
    of the plugin, as is illustrated below for the `field.Painter` parameter:
    
.. literalinclude:: ../examples/power/test_cross_power.params
    :lines: 11-12  
    
For more examples on how to accurately represent plugins in configuration files, 
see the myriad of configuration files listed in the ``examples`` directory
of the source code.

Specifying the output file
~~~~~~~~~~~~~~~~~~~~~~~~~~

All configuration files must include the ``output`` parameter. This parameter
gives the name of the output file to which the results of the algorithm will be saved. 

The ``nbkit.py`` script will raise an exception when the ``output`` parameter is not 
present in the input configuration file.

Specifying the cosmology
~~~~~~~~~~~~~~~~~~~~~~~~

For the succesful reading of data using some nbodykit DataSource classes, cosmological parameters
must be specified. The desired cosmology should be set in the configuration file, as is done in line
4 of the previous example. A single, global cosmology class will be initialized and passed to all 
DataSource objects that are created while running the nbodykit code. 

The cosmology class is located at :class:`nbodykit.cosmology.Cosmology`, and the
syntax for the class is borrowed from :class:`astropy.cosmology.wCDM` class. The constructor
arguments are:

.. ipython:: python

    from nbodykit.cosmology import Cosmology
    
    # can also do ``Cosmology?`` in IPython
    help(Cosmology.__init__)
    

Running in batch mode
---------------------

The nbodykit code also provides a tool to run a specific Algorithm for a set of configuration 
files, possibly executing the algorithms in parallel. We refer to this as "batch mode" and 
provide the `nbkit-batch.py`_ script in the ``bin`` directory for this purpose.


Once again, the ``-h`` flag will provide the help message for this script; the intended
usage is:

.. code-block:: bash

    mpirun -n [n] python bin/nbkit-batch.py [--extras EXTRAS] [--debug] [--use_all_cpus] -i TASKS -c CONFIG  AlgorithmName cpus_per_worker


The idea here is that a "template" configuration file can be passed to ``nbkit-batch.py`` via the ``-c`` option,
and this file should contain special keys that will be formatted using :meth:`str.format` syntax when iterating
through a set of configuration files. The names of these keys and the desired values for the 
keys to take when iterating can be specified by the ``-i`` option. 

By example
~~~~~~~~~~

Let's consider the following invocation of the ``nbkit-batch.py`` script:

.. code-block:: bash
    
    mpirun -np 7 python bin/nbkit-batch.py FFTPower 2 -c examples/batch/test_power_batch.template -i "los: [x, y, z]" --extras examples/batch/extra.template
    
In this example, the code is executed using MPI with 7 available processors, and we have set `cpus_per_worker` to 2. 
The ``nbkit-batch.py`` script reserves one processor to keep track of the task scheduling (the "master" processor), 
which means that 6 processors
are available for computation. With 2 cpus for each worker, the script is able to use 3 workers to execute `FFTPower` 
algorithms in parallel. Furthermore, we have asked for 3 task values -- the input configuration template will have the 
`los` key updated with values 'x', 'y', and 'z'. With only three tasks and exactly 3 workers, each task can be computed 
in parallel simulataneously.

For a closer look at how the task values are updated in the template configuration file, let's examine the template
file:

.. literalinclude:: ../examples/batch/test_power_batch.template

In this file, we see that there is exactly one task key: `los`. The ``{los}`` string will be updated with the values
given on the command-line ('x', 'y', and 'z'), and the `FFTPower` algorithm will be executed for each of the resulting
configuration files. The task keys are formatted using the Python string formatting syntax of :meth:`str.format`.

Lastly, we have also passed a file to the ``nbkit-batch.py`` script using the ``--extras`` option. This option allows an arbitrary
number of extra string keys to be formatted for each task iteration. In this example, the only "extra" key provided 
is ``{tag}``, and the ``extra.template`` file looks like:

.. literalinclude:: ../examples/batch/extra.template

So, when updating `los` to the first task value ('x'), the `tag` key is updated to 'task_1', and the pattern continues
for the other tasks. With this configuration, ``nbkit-batch.py`` will output 3 separate files, named:

    - test_batch_power_fastpm_1d_xlos_task_1.dat
    - test_batch_power_fastpm_1d_ylos_task_2.dat
    - test_batch_power_fastpm_1d_zlos_task_3.dat

Multiple task keys
~~~~~~~~~~~~~~~~~~

The ``-i`` flag can be passed multiple times to the ``nbkit-batch.py`` script. For example, let us imagine that
in addition to the `los` task key, we also wanted to iterate over a `box` key. If we had two boxes, labeled `1`
and `2`, then we could also specify ``-i box: ['1', '2']`` on the command-line. Then, the task values that would be iterated over are::

(`los`, `box`) = ('x', '1'), ('x', '2'), ('y', '1'), ('y', '2'), ('z', '1'), ('z', '2')






.. _nbkit.py: https://github.com/bccp/nbodykit/tree/master/bin/nbkit.py
.. _nbkit-batch.py: https://github.com/bccp/nbodykit/tree/master/bin/nbkit-batch.py
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _`YAML`: http://pyyaml.org/wiki/PyYAMLDocumentation
