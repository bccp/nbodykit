Benchmarking nbodykit on NERSC
==============================

Here, we provide scripts to run benchmarking tests of various nbodykit algorithms on NERSC
in a reproducible manner. The ``run.py`` scripts in the algorithm directories provide
the user with a set of commands that will submit the necessary job to NERSC for different
configurations. To see the list of available commands, use

.. code:: bash
   
   $ python run.py -i

Passing the integer value of the command via the command-line will create the necessary job script
and submit the job to NERSC, printing the job script to stdout:

.. code:: bash
   
   $ python run.py 0


Any command-line option supported by ``benchmark.py`` can also be specified here. For example, to run 
the first command using Python version 2.7, submitting the job to the regular queue with a 90 minute allocation, use:

.. code:: bash

   $ python run.py 0 -p regular -t 90 --py 2.7
