nbodykit
========

Software kit for cosmological N-body simulations. 


Build Status
------------

The CI tests are run in a miniconda environment.

.. image:: https://api.travis-ci.org/bccp/nbodykit.svg
    :alt: Build Status
    :target: https://travis-ci.org/bccp/nbodykit/



Top Level Executables
=====================

The tools are provide as top level executable scripts. 
All tools are MPI parallel.  We can deal with very large simulations.

- ``bin/power.py`` is a Power Spectrum calculator, which also calculates correlation functions
  via a Fast Fourier Transform.

- ``bin/corr.py`` is a point-point pair-counting based Correlation Function calculator.

- ``bin/fof.py`` is a friend of friend feature (halo) identifier.

- ``bin/subhalo.py`` is a FOF6D sub-feature (subhalo) identifier .

- ``bin/power-parallel.py`` schedules and runs a set of ``power.py`` jobs, possibly using several independent nodes

Run them with '-h' to see the inline help.

Each takes a few 'datasource' strings that specify the format of the inputs; most are documented in '-h' as well.
Additional datasource plugins can be added with '-X' commandline arguments. Some examples are in contrib/ directory.

Examples
--------

There are example scripts (which also act as integrated tests) in examples directory.
The supporting data for these scripts can be retrieved from 

    https://s3-us-west-1.amazonaws.com/nbodykit/nbodykit-data.tar.gz

Check get_data.sh for details.

Dependencies
------------

.. _`pfft-python`: http://github.com/rainwoodman/pfft-python
.. _`pfft`: http://github.com/mpip/pfft
.. _`pmesh`: http://github.com/rainwoodman/pmesh
.. _`kdcount`: http://github.com/rainwoodman/kdcount
.. _`sharedmem`: http://github.com/rainwoodman/sharedmem
.. _`MP-sort`: http://github.com/rainwoodman/MP-sort
.. _`h5py`: http://github.com/h5py/h5py

The software is built on top of existing tools. Please refer to their
documentations:

- `pfft`_    : massively parallel fast fourier transform with pencil domains
- `pfft-python`_  : python binding of pfft
- `pmesh`_     :  particle mesh framework in Python
- `kdcount`_   : pair-counting and friend-of-friend clustering with KD-Tree
- `MP-sort`_   : massively parallel sorting 
- `sharedmem`_ : in-node parallelism with fork and copy-on-write.

Some better established dependencies are

- `h5py`_     : input and output of HDF5 files
- `mpi4py`_     : MPI for python
- `scipy`_     
- `numpy`_     

Optional Dependencies
---------------------

- pandas and pytables are required to access the PANDAS subset of HDF5 
  and fast parsing of plain text files.

Build
-----

The software is designed to be installed with the ``pip`` utility. The recommended setup is to install 
the software in 'developer mode' via the ``-e`` option of ``pip``. To install the main ``nbodykit`` package 
in 'developer mode', as well as the external dependencies listed above, into the default python installation 
directory use:

.. code:: sh
   
    git clone http://github.com/bccp/nbodykit
    cd nbodykit

    pip install -r requirements.txt -e .

A different installation directory can be specified via the ``--user`` or ``--root <dir>`` 
options of the ``pip install`` command.

The pure-python ``nbodykit`` package (without external dependencies) can be installed by 
omitting the ``-r requirements.txt`` option, with such an installation only requiring ``numpy``. 
The caveat being that the functionality of the package is greatly diminished -- package behavior 
in this instance is not tested and considered undefined. 

Please note that there are slight changes to the above procedure on Mac and Edison, 
as explictly noted below.

The dependencies of nbodykit are not fully stable, thus we recommend updating
the external dependencies occassionally via the ``-U`` option of ``pip install``:

.. code:: sh

    # It may take a while to build fftw and pfft.
    # Mac and Edison are special, see notes below

    pip install -U -r requirements.txt


Now we shall be able to use nbodykit, in a interactive python session 
(please remember to jump to bin/ directory to avoid weird issues about importing in-tree)

.. code:: python

    import nbodykit
    print(nbodykit)

    import kdcount
    print(kdcount)

    import pmesh
    print(pmesh)

Or run the scripts in the bin directory:

.. code:: bash

    cd bin/
    mpirun -n 4 python-mpi fof.py -h


Special instructions for Mac and Edison
---------------------------------------

Mac Notes
+++++++++

autotools are needed on a Mac

.. code::

    sudo port install autoconf automake libtool
    
On Mac, the `LDSHARED` environment variable must be explicitly set. In bash, the command is

.. code::

    export LDSHARED="mpicc -bundle -undefined dynamic_lookup -DOMPI_IMPORTS"; pip install -r requirements.txt -e .
    
On recent versions of MacPorts, we also need to tell mpicc to use gcc rather than the default clang
compiler, which doesn't compile fftw correctly due to lack of openmp support.

.. code::
    
    export OMPI_CC=gcc
 
Edison/Cori Notes
++++++++++++

On Edison and Cori, the recommended python distribution is anaconda. 
If darshan or altd are loaded by default, be sure to unload them to avoid issues:

.. code::

    module unload darshan
    module unload altd

and preferentially, load PrgEnv-gnu

.. code::

    module unload PrgEnv-intel
    module unload PrgEnv-gray
    module load PrgEnv-gnu

then load python

.. code::

    module load python/2.7-anaconda

Not absolutely necessary, but it is wise to set up the conda environment in a faster file system.
Modify .condarc to add a line like this

.. code::

    changeps1: false
    envs_dirs :
        - /project/projectdirs/{your directory on project}/envs

Then, you can create a new anaconda environment to install ``nbodykit`` and 
its dependencies by cloning the default anaconda environment:

.. code::
    
    conda create -n myenv --clone root
    source activate myenv

To speed up calculations with the fast python-mpi launcher in
/project/projectdirs/m779/python-mpi, 
we can tar the anaconda environment via

.. code:: bash

    bash /project/projectdirs/m779/python-mpi/tar-anaconda.sh 
            /project/projectdirs/{your directory on project}/myenv.tar.gz \
            /project/projectdirs/{your directory on project}/envs/myenv

To install ``nbodykit`` and its dependencies into 'myenv', use:

.. code::
    
    MPICC=cc pip install -r requirements.txt
    pip install -e .

And also tar nbodykit with its requirements for compute-nodes

.. code:: bash

    bash /project/projectdirs/m779/python-mpi/tar-pip.sh nbodykit.tar.gz -r requirements.txt .

And example job script on Cori is

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

