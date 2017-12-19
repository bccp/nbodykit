Install nbodykit
================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. _getting-nbodykit:

Getting nbodykit
----------------


nbodykit is currently supported on macOS and Linux architectures. The
recommended installation method is to install nbodykit and its
dependencies as part of the `Anaconda <https://www.continuum.io/downloads>`_
Python distribution. For more advanced users, or those without an
Anaconda distribution, the software can also be installed using the ``pip``
utility or directly from the source. In these latter cases, additional
difficulties may arise when trying to compile and install some of
nbodykit's dependencies.

The package is available for Python versions 2.7, 3.5, and 3.6.

.. _conda-installation:

Installing nbodykit with Anaconda
---------------------------------

The easiest installation method uses the ``conda`` utility, as part
of the `Anaconda <https://www.continuum.io/downloads>`_ package
manager. The distribution can be downloaded and installed for free from
https://www.continuum.io/downloads.

We have pre-built binaries for nbodykit and all of its dependencies available
via Anaconda that are compatible with Linux and macOS platforms.
To avoid dependency conflicts, we recommend installing nbodykit into a
fresh environment. This can be achieved with the following commands:

.. code-block:: bash

  $ conda create --name nbodykit-env python=3 # or use python=2 for python 2.7*
  $ source activate nbodykit-env
  $ conda install -c bccp nbodykit

Updating nbodykit and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To keep nbodykit and its dependencies up to date, use

.. code-block:: bash

  $ conda update -c bccp --all

Installing From Source for Conda-based Installs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can also install nbodykit from source from within a conda environment,
given that ``git`` is installed. First, clone the GitHub repository,

.. code-block:: bash

  $ git clone http://github.com/bccp/nbodykit
  $ cd nbodykit

Then, install all of the required dependencies using

.. code-block:: bash

  $ conda install -c bccp --file requirements.txt
  $ conda install -c bccp --file requirements-extras.txt

Now, the desired git branch can be installed easily. By default, the ``master``
branch is active, and the latest development state of nbodykit can be installed
using

.. code-block:: bash

  $ pip install -e .

Note that the ``-e`` flag installs nbodykit in "develop" mode, allowing the
installed version to reflect the latest changes in the source code. The latest
changes to the ``master`` branch can be incorporated from GitHub via

.. code-block:: bash

  $ git checkout master
  $ git pull origin master

.. _pip-installation:

Installing nbodykit with ``pip``
--------------------------------

.. warning::

    The easiest and recommended method to install nbodykit and its dependencies
    is using the Anaconda package. See :ref:`conda-installation` for more details.

To build nbodykit from source, you will need to make sure all of the dependencies
are properly installed on your system. To start, the following dependencies
should be installed first:

.. code-block:: bash

    $ pip install numpy cython mpi4py

Next, we must compile the remaining dependencies, which depends on the user's
machine.

Linux
~~~~~

To install nbodykit as well as all of its external dependencies on a Linux machine
into the default Python installation directory:

.. code-block:: bash

    $ pip install nbodykit[extras]

A different installation directory can be specified via the ``--user`` or
``--root <dir>`` options of the ``pip install`` command.

macOS
~~~~~

More care is required to properly build the dependencies on macOS machines.
The ``autotools`` software is required, which can be installed using
the `MacPorts <https://www.macports.org/install.php>`_ package manager using:

.. code-block:: bash

    $ sudo port install autoconf automake libtool

Using recent versions of MacPorts, we also need to tell ``mpicc`` to use ``gcc``
rather than the default ``clang`` compiler, which doesn't compile ``fftw`` correctly
due to the lack of ``openmp`` support. Additionally, the ``LDSHARED``
environment variable must be explicitly set.

In bash, the installation command is:

.. code-block:: bash

    $ export OMPI_CC=gcc
    $ export LDSHARED="mpicc -bundle -undefined dynamic_lookup -DOMPI_IMPORTS"; pip install nbodykit[extras]

This command will compile and install the dependencies of nbodykit and then
install nbodykit. Again, a different installation directory can be specified via
the ``--user`` or ``--root <dir>`` options of the ``pip install`` command.

.. _nbodykit-on-NERSC:

nbodykit on NERSC
-----------------

.. note::

    This section covers using nbodykit on the computing nodes of NERSC.
    The computing nodes requires special care because they do not work with
    the simple MPI provided from Anaconda.


    If instead you wish to use nbodykit on the login nodes of NERSC or the
    Jupyter Hub services (available at https://jupyter.nersc.gov and
    https://jupyter-dev.nersc.gov/), users should follow the
    :ref:`Anaconda installation instructions <conda-installation>`
    to install nbodykit. The login nodes and JuptyerHub machines are very
    similar to standard computers. For more information on the JupyterHub
    services, see `the official NERSC guide`_.

Development and testing of nbodykit was performed on the `NERSC`_ super-computing
machines at Lawrence Berkeley National Laboratory. We maintain a daily build of
the latest stable version of nbodykit on NERSC systems for Python versions
2.7, 3.5, and 3.6 and provide a tool to automatically load
the appropriate environment when running jobs on either the `Edison`_ or `Cori`_
machines.

To load the latest stable version of nbodykit on NERSC, the following line
should be added to the beginning of the user's job script:

.. code-block:: bash

  # load python 3.6 with latest stable nbodykit
  # can also specify 2.7 or 3.5 here
  source /usr/common/contrib/bccp/conda-activate.sh 3.6


If instead the user wishes to install the latest development version
of nbodykit, the following lines should be added to the job script:

.. code-block:: bash

  # first load python 3.6 with latest stable nbodykit
  # can also specify 2.7 or 3.5 here
  source /usr/common/contrib/bccp/conda-activate.sh 3.6

  # overwrite nbodykit with the latest version from the tip of master
  bcast-pip git+git://github.com/bccp/nbodykit.git

In the nbodykit source directory, we include an example Python script
and job script for users. To run this example on NERSC, first download
the necessary files:

.. code-block:: bash

  # download the example script
  $ wget https://raw.githubusercontent.com/bccp/nbodykit/master/nersc/example.py

  # download the job script
  $ wget https://raw.githubusercontent.com/bccp/nbodykit/master/nersc/example-job.slurm

and then if on the Cori machine, the job can be submitted using

.. code-block:: bash

  $ sbatch -C haswell example-job.slurm

of if on the Edison machine, use

.. code-block:: bash

  $ sbatch example-job.slurm

The example job script simply loads the nbodykit environment and executes
the Python script in parallel, in this case, using 16 CPUs.

.. code-block:: bash

    #!/bin/bash
    #SBATCH -p debug
    #SBATCH -o nbkit-example
    #SBATCH -n 16

    # load nbodykit
    source /usr/common/contrib/bccp/conda-activate.sh 3.6

    # run the main nbodykit example
    srun -n 16 python example.py

If successful, this will save a file ``nbkit_example_power.json`` to the
current working directory.

.. _`NERSC`: http://www.nersc.gov/systems/
.. _`Edison`: https://www.nersc.gov/users/computational-systems/edison/
.. _`Cori`: https://www.nersc.gov/users/computational-systems/cori
.. _`the official NERSC guide`: http://www.nersc.gov/users/data-analytics/data-analytics-2/jupyter-and-rstudio/
