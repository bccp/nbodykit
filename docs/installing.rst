Installation
============

Required dependencies
---------------------

The well-established dependencies are: 

- Python 2.7 or 3.4
- `scipy`_,  `numpy`_   : the foundations for scientific Python
- `mpi4py`_   : MPI for Python
- `h5py`_     : support for HDF5 files in Python

with a suite of additional tools: 

- `astropy`_ : a community Python library for astronomy
- `pfft-python`_  : a Python binding of `pfft`_, a massively parallel Fast Fourier Transform implementation with pencil domains
- `pmesh`_     :  a particle mesh framework in Python
- `kdcount`_   : pair-counting and Friends-of-Friends clustering with KD-Tree
- `bigfile`_   :  a reproducible, massively parallel IO library for hierarchical data
- `MP-sort`_   : massively parallel sorting 
- `sharedmem`_ : in-node parallelism with fork and copy-on-write

.. _astropy: https://astropy.readthedocs.io/en/stable/
.. _scipy: http://github.com/scipy/scipy
.. _numpy: http://github.com/numpy/numpy
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _h5py: http://github.com/h5py/h5py
.. _pfft: http://github.com/mpip/pfft
.. _pfft-python: http://github.com/rainwoodman/pfft-python
.. _pmesh: http://github.com/rainwoodman/pmesh
.. _kdcount: http://github.com/rainwoodman/kdcount
.. _bigfile: https://github.com/rainwoodman/bigfile
.. _MP-sort: http://github.com/rainwoodman/MP-sort
.. _sharedmem: http://github.com/rainwoodman/sharedmem

Optional dependencies
---------------------

For reading data using pandas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `pandas`_ and `pytables`_ are required to use the :class:`~nbodykit.plugins.datasource.Pandas` DataSource, which uses `pandas` for fast parsing of plain text files, as well as the `pandas` subset of HDF5

.. _pandas: http://pandas.pydata.org/
.. _pytables: http://www.pytables.org

For creating data using a Halo Occupation Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `halotools`_ is required to use the :class:`~nbodykit.plugins.datasource.Zheng07HOD` DataSource, which provides a general framework for populating halos with galaxies using Halo Occupation Distribution modeling

.. _halotools: http://halotools.readthedocs.io/en/latest/

For generating simulated data from linear power spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `classylss`_, a python binding of the Boltzmann code `CLASS`_, is required to use the :class:`~nbodykit.plugins.datasource.ZeldovichSim` DataSource, which computes a simulated data catalog using the Zel'dovich approximation (from a linear power spectrum computed with CLASS)

.. _classylss: https://github.com/nickhand/classylss
.. _CLASS: https://github.com/lesgourg/class_public

Instructions
------------

The software is designed to be installed with the ``pip`` utility like a regular
Python package. The first step on all supported platforms is to checkout the 
source code via: 

.. code-block:: sh
   
    git clone http://github.com/bccp/nbodykit
    cd nbodykit
    
Linux
~~~~~

The steps listed below are intended for a commodity Linux-based cluster 
(e.g., a `Rocks cluster`_) or a Linux-based workstation / laptop.

To install the main nbodykit package, as well as the external dependencies 
listed above, into the default Python installation directory:

.. code-block:: sh
   
    pip install -r requirements.txt
    pip install -U --force --no-deps .

A different installation directory can be specified via the ``--user`` or ``--root <dir>`` 
options of the ``pip install`` command. 

Mac OS X
~~~~~~~~

The ``autotools`` software is needed on Mac:

.. code-block:: bash

    sudo port install autoconf automake libtool
    
Using recent versions of MacPorts, we also need to tell ``mpicc`` to use ``gcc`` 
rather than the default ``clang`` compiler, which doesn't compile ``fftw`` correctly 
due to the lack of ``openmp`` support. Additionally, the ``LDSHARED`` 
environment variable must be explicitly set. 

In bash, the installation command is:

.. code-block:: bash
    
    export OMPI_CC=gcc
    export LDSHARED="mpicc -bundle -undefined dynamic_lookup -DOMPI_IMPORTS"; pip install -r requirements.txt 
    pip install -U --force --no-deps .
    
Development Mode
~~~~~~~~~~~~~~~~~

nbodykit can be installed with the development mode (``-e``) of pip

.. code-block:: bash

    pip install -r requirements.txt -e .

In addition to the dependency packages, the 'development' installation
of nbodykit may require a forced update from time to time:

.. code-block:: bash

    pip install -U --force --no-deps -e .

It is sometimes required to manually remove the ``nbodykit`` directory in 
``site-packages``, if the above command does not appear to update the installation
as expected.

Final Notes
~~~~~~~~~~~

The dependencies of nbodykit are not fully stable, thus we recommend updating
the external dependencies occassionally via the ``-U`` option of ``pip install``. 
Also, the ``--force`` option ensures that the current sourced version is installed:

.. code-block:: bash

    pip install -U -r requirements.txt
    pip install -U --force --no-deps .

To confirm that nbodykit is working, we can type, in a interactive Python session:

.. code-block:: python

    import nbodykit
    print(nbodykit)

    import kdcount
    print(kdcount)

    import pmesh
    print(pmesh)

Or try the scripts in the bin directory:

.. code-block:: bash

    cd bin/
    mpirun -n 4 python nbkit.py -h
    
To run the test suite after installing nbodykit, install `py.test`_ and 
`pytest-pipeline`_ and run ``py.test nbodykit`` from the base directory
of the source code:

.. code-block:: bash
    
    pip install pytest pytest-pipeline
    pytest nbodykit

.. _`Rocks Cluster`: http://www.rocksclusters.org/rocks-documentation/4.3/
.. _`py.test`: http://pytest.org/latest/
.. _`pytest-pipeline`: `https://pypi.python.org/pypi/pytest-pipeline`

 
.. _nbodykit-on-nersc:

Using nbodykit on NERSC
-----------------------

In this section, we give instructions for using the latest stable build of nbodykit on `NERSC`_  
machines (`Edison`_ and `Cori`_), which is provided ready-to-use and is recommended for first-time
users. For more advanced users, we also provide instructions for performing active development of the 
source code on NERSC.

When using nbodykit on NERSC, we need to ensure that the Python environment is set up to work 
efficiently on the computing nodes. The default Python start-up time scales badly with the number 
of processes, so we employ the `python-mpi-bcast`_ tool to ensure fast and reliable start-up times 
when using nbodykit. This tool can be accessed on both the Cori and Edison machines. 

General Usage
~~~~~~~~~~~~~

We maintain a daily build of the latest stable version of nbodykit on NERSC systems 
that works with the ``2.7-anaconda`` Python module and uses the `python-mpi-bcast` helper 
tool for fast startup of Python. Please see `this tutorial`_ for further details about 
using `python-mpi-bcast` to launch Python applications on NERSC.

In addition to up-to-date builds of nbodykit, we provide a tool 
(``/usr/common/contrib/bccp/nbodykit/activate.sh``) designed to be used in job scripts to automatically
load nbodykit and ensure a fast startup time using `python-mpi-bcast`.

Below is an example job script that prints the help message of the 
:class:`FFTPower <nbodykit.plugins.algorithms.PeriodicBox.FFTPowerAlgorithm>` algorithm:

.. literalinclude:: ../nersc/example.sh
    :language: bash

.. _`NERSC`: http://www.nersc.gov/systems/
.. _`this tutorial`: https://github.com/rainwoodman/python-mpi-bcast/wiki/NERSC

Active development
~~~~~~~~~~~~~~~~~~
If you would like to use your own development version of nbodykit directly on NERSC, 
more installation work is required, although we also provide tools to simplify this process.

We can divide the addititional work into 3 separate steps:

1. When building nbodykit on a NERSC machine, we need to ensure the Python environment
is set up to work efficiently on the computing nodes.

If `darshan`_ or `altd`_ are loaded by default, be sure to unload them before installing, 
as they tend to interfere with Python:

.. code-block:: bash

    module unload darshan
    module unload altd

and preferentially, use GNU compilers from PrgEnv-gnu

.. code-block:: bash

    module unload PrgEnv-intel
    module unload PrgEnv-cray
    module load PrgEnv-gnu

then load the `Anaconda`_ Python distribution,

.. code-block:: bash

    module load python/2.7-anaconda

For convenience, these lines can be included in the shell profile configuration 
file on NERSC (i.e., ``~/.bash_profile.ext``).

2. For easy loading of nbodykit on the compute nodes, we provide tools to create 
separate bundles (tarballs) of the nbodykit source code and dependencies.
This can be performed using the ``build.sh`` script in the ``nbodykit/nersc``
directory in the source code tree. 

.. code-block:: bash

    cd nbodykit/nersc;

    # build the dependencies into a bundle
    # this creates the file `$NERSC_HOST/nbodykit-dep.tar.gz`
    bash build.sh deps
    
    # build the source code into a separate bundle
    # this creates the file `$NERSC_HOST/nbodykit.tar.gz`
    bash build.sh source
    
When the source code changes or the dependencies need to be updated, 
simply repeat the relevant ``build.sh`` command given above to regenerate the
bundle.

3. Finally, in the job script, we must explicitly activate `python-mpi-bcast`
and load the nbodykit bundles. 

.. code-block:: bash

    #!/bin/bash
    #SBATCH -p debug
    #SBATCH -o nbkit-dev-example
    #SBATCH -n 16
    
    # load anaconda
    module unload python
    module load python/2.7-anaconda
    
    # activate python-mpi-bcast
    source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh
    
    # go to the nbodykit source directory
    cd /path/to/nbodykit
    
    # bcast the nbodykit tarballs
    bcast nersc/$NERSC_HOST/nbodykit-dep.tar.gz nersc/$NERSC_HOST/nbodykit.tar.gz
    
    # run the main nbodykit executable
    srun -n 16 python-mpi /dev/shm/local/bin/nbkit.py FFTPower --help


.. _`Edison`: https://www.nersc.gov/users/computational-systems/edison/
.. _`Cori`: https://www.nersc.gov/users/computational-systems/cori
.. _`darshan`: http://www.mcs.anl.gov/research/projects/darshan/
.. _`altd`: http://www.nersc.gov/users/software/programming-libraries/altd/
.. _`Anaconda`: http://docs.continuum.io/anaconda/index
.. _`python-mpi-bcast`: https://github.com/rainwoodman/python-mpi-bcast