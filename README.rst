nbodykit
========

nbodykit is a software kit for cosmological datasets from
N-body simulations and large scale structure surveys.

Driven by the optimistism regarding the abundance and availability of 
large scale computing resources in the future, 
the development of nbodykit
distinguishes itself from other similar software packages
([nbodyshop]_, [pynbody]_, [yt]_, [xi]_) by focusing on :

- a **unified** treatment of simulation and observational data sets; 
  insulating algorithms from data containers;

- reducing wall-clock time by **scaling** to thousands of cores;

- **deployment** and availability on large super computing facilities.

All algorithms are parallel, and run with Message Passing Interface (MPI).

Build Status
------------

We perform integrated tests for the toplevel executables in a
miniconda environment for Python 2.7 and Python 3.4. 

.. image:: https://api.travis-ci.org/bccp/nbodykit.svg
    :alt: Build Status
    :target: https://travis-ci.org/bccp/nbodykit/


Top Level Executables
---------------------

The algorithms implemented in nbodykit can be invoked by

- ``bin/nbkit.py`` 

- ``bin/nbkit-batch.py`` 

Run them with '-h' to see the inline help.

.. code:: bash

    mpirun -n 4 python bin/nbkit.py Algorithm -o output_location [ .... ]


Algorithms
----------

Algorithms are implemented as Plugins.

- FFTPower : Power spectrum (Isotropic, Anisotropic, and multipoles) via Fast Fourier Transforms

- FFTCorrelation: Correlation function (Isotropic, Anisotropic, and multipoles) via Fast Fourier Transforms

- PairCountCorrelation: Correlation function (Isotropic, Anisotropic, and multipoles) via simple pair-counting

- FOF : Friend of Friend halos (usually called groups)

- FOF6D : 6D Friend of Friend halos (usually called subhalos or galaxies)

- Describe : Describes the max and min of a column from any DataSource

- Subsample : Create a subsample from a DataSource, and evaluate density (:math:`1 + \delta`) smoothed 
  at the given scale

- TraceHalo : Trace the center of mass and velocity of particles between two different DataSource, joining
  by ID.

Examples
--------

There are example scripts (which also act as integrated tests) in examples directory.
The supporting data for these scripts can be retrieved from 

    https://s3-us-west-1.amazonaws.com/nbodykit/nbodykit-data.tar.gz

Check get_data.sh for details.

Dependencies
------------

The software is built on top of existing tools. Please refer to their
documentations:

- [pfft]_    : massively parallel fast fourier transform with pencil domains
- [pfft-python]_  : python binding of pfft
- [pmesh]_     :  particle mesh framework in Python
- [kdcount]_   : pair-counting and friend-of-friend clustering with KD-Tree
- [bigfile]_   :  A reproducible massively parallel IO library for hierarchical data
- [MP-sort]_   : massively parallel sorting 
- [sharedmem]_ : in-node parallelism with fork and copy-on-write.

Some better established dependencies are

- [scipy]_,  [numpy]_   : the foundations for Scientific Python.
- [mpi4py]_   : MPI for python
- [h5py]_     : Support for HDF5 files

Optional Dependencies
---------------------

- [pandas]_, [pytables]_ are required to access the PANDAS subset of HDF5 and fast parsing of plain text files.

Build
-----

The software is designed to be installed with the ``pip`` utility like a regular
python package.

Using nbodykit from the source tree is not supported. See 'Development mode' for
details.

The steps listed here is intended for a commodity Linux based cluster 
(e.g. a Rocks Cluster [rocksclusters]_) or a Linux based workstation / laptop.
Please note that there are slight changes to the procedure on systems running
a Mac OS X operating system and 
Cray super-computers 
as explictly noted below in `Special notes for Mac and Cray`_.

Install the main ``nbodykit`` package, as well as the external dependencies 
listed above, into the default python installation directory with:

.. code:: sh
   
    git clone http://github.com/bccp/nbodykit
    cd nbodykit

    # It may take a while to build fftw and pfft.
    # Mac and Edison are special, see notes below

    pip install -r requirements.txt
    pip install -U --force --no-deps .

A different installation directory can be specified via the ``--user`` or ``--root <dir>`` 
options of the ``pip install`` command. 

The pure-python ``nbodykit`` package (without external dependencies) can be installed by 
omitting the ``-r requirements.txt`` option, with such an installation only requiring ``numpy``. 
The caveat being that the functionality of the package is greatly diminished -- package behavior 
in this instance is not tested and considered undefined. 


The dependencies of nbodykit are not fully stable, thus we recommend updating
the external dependencies occassionally via the ``-U`` option of ``pip install``. 
Also, since nbodykit is
not yet stable enough for versioned releases, ``--force`` ensures the current 
sourced version is installed:

.. code:: sh

    pip install -U -r requirements.txt
    pip install -U --force --no-deps .

To confirm that nbodykit is working, we can type, in a interactive python session:
(please remember to jump to bin/ directory to avoid weird issues about importing in-tree)

.. code:: python

    import nbodykit
    print(nbodykit)

    import kdcount
    print(kdcount)

    import pmesh
    print(pmesh)

Or try the scripts in the bin directory:

.. code:: bash

    cd bin/
    mpirun -n 4 python-mpi fof.py -h

Development Mode
++++++++++++++++

nbodykit can be installed with the development mode (``-e``) of pip

.. code::

    pip install -r requirements.txt -e .

In addition to the dependency packages, the 'development' installation
of nbodykit may require a forced update from time to time:

.. code::

    pip install -U --force --no-deps -e .

It is sometimes required to manually remove the ``nbodykit`` directory in 
``site-packages``, if the above command does not appear to update the installation
as expected.


Special notes for Mac and Cray
------------------------------

Mac Notes
+++++++++

autotools are needed on a Mac

.. code::

    sudo port install autoconf automake libtool
    
On Mac, the `LDSHARED` environment variable must be explicitly set. In bash, the command is

.. code::

    export LDSHARED="mpicc -bundle -undefined dynamic_lookup -DOMPI_IMPORTS"; pip install -r requirements.txt .
    
On recent versions of MacPorts, we also need to tell mpicc to use gcc rather than the default clang
compiler, which doesn't compile fftw correctly due to lack of openmp support.

.. code::
    
    export OMPI_CC=gcc
 
Edison/Cori Notes
+++++++++++++++++

To use nbodykit on a Cray system (e.g. [Edison]_, [Cori]_), we need to ensure the python environment
is set-up to working efficiently on the computing nodes.

If darshan [darshan]_ or altd are loaded by default, be sure to unload them since they tend to interfere
with Python:

.. code::

    module unload darshan
    module unload altd

and preferentially, use GNU compilers from PrgEnv-gnu

.. code::

    module unload PrgEnv-intel
    module unload PrgEnv-cray
    module load PrgEnv-gnu

then load the Anaconda [anaconda]_ python distribution,

.. code::

    module load python/2.7-anaconda

We will need to set up the fast python start-up on a Cray computer, since
the default start-up scales badly with the number of processes. We lay out
the procedures below:

1. modify .condarc to add a line like this, to move the default
a python environment from /home to /project, a faster file system

.. code::

    changeps1: false
    envs_dirs :
        - /project/projectdirs/{your directory on project}/envs

2. Create a new anaconda environment (to receive updates). 

.. code::
    
    conda create -n myenv --clone root
    source activate myenv
    # optionally update to a more recent version of the packages.
    conda update
    
3.  Create a bundle (tarball) of the environment for the fast python-mpi launcher [fast-python]_ in
`/project/projectdirs/m779/python-mpi`, 

.. code:: bash

    bash /project/projectdirs/m779/python-mpi/tar-anaconda.sh 
            /project/projectdirs/{your directory on project}/myenv.tar.gz \
            /project/projectdirs/{your directory on project}/envs/myenv

4.  Create a bundle (tarball) of nbodykit, as well. Repeat this step if nbodykit or any
dependency is updated.

.. code:: bash

    MPICC=cc bash /project/projectdirs/m779/python-mpi/tar-pip.sh nbodykit.tar.gz -r requirements.txt .

After these steps we can use nbodykit with a job script similar to the example below.
Notice that there is no need to install nbodykit to the newly created python
environment, the computing nodes will use the software in the bundle file ``nbodykit.tar.gz`` we created
in step 4.

.. code:: bash

    #! /bin/bash
    #SBATCH -o 40steps-pm-79678.powermh.%j
    #SBATCH -N 16
    #SBATCH -p debug
    #SBATCH -t 00:30:00
    #SBATCH -J 40steps-pm-79678.powermh

    set -x

    export OMP_NUM_THREADS=1
    export ATP_ENABLED=0
    source /project/projectdirs/m779/python-mpi/activate.sh /dev/shm/local "srun -n 512"

    bcast -v {your projectdir}/myenv.tar.gz
    bcast -v nbodykit.tar.gz

    srun -n 512 python-mpi \
    {your nbodykit dir}/bin/power.py \
    2d 2048 power2d_40steps-pm_mh14.00_1.0000.txt \
    TPMSnapshot:$SCRATCH/crosshalo/40steps-pm/snp00100_1.0000.bin:1380:-rsd=z \
    FOFGroups:fof00100_0.200_1.0000.hdf5:1380:2.4791e10:"-select=Rank < 79678":-rsd=z


References
==========

.. [nbodyshop] http://www-hpcc.astro.washington.edu/tools/tools.html

.. [pynbody] https://github.com/pynbody/pynbody

.. [yt] http://yt-project.org/
    
.. [pfft-python] http://github.com/rainwoodman/pfft-python

.. [pfft] http://github.com/mpip/pfft

.. [pmesh] http://github.com/rainwoodman/pmesh

.. [kdcount] http://github.com/rainwoodman/kdcount

.. [sharedmem] http://github.com/rainwoodman/sharedmem

.. [MP-sort] http://github.com/rainwoodman/MP-sort

.. [h5py] http://github.com/h5py/h5py

.. [numpy] http://github.com/numpy/numpy

.. [scipy] http://github.com/scipy/scipy

.. [pandas] http://pandas.pydata.org/

.. [pytables] http://pandas.pydata.org/

.. [mpi4py] https://bitbucket.org/mpi4py/mpi4py

.. [fast-python] https://github.com/rainwoodman/python-mpi-bcast

.. [bigfile] https://github.com/rainwoodman/bigfile

.. [rocksclusters] http://rocksclusters.org

.. [xi] http://github.com/bareid/xi

.. [edison] https://www.nersc.gov/users/computational-systems/edison/

.. [cori] https://www.nersc.gov/users/computational-systems/cori/

.. [darshan] http://www.mcs.anl.gov/research/projects/darshan/

.. [anaconda] http://docs.continuum.io/anaconda/index

