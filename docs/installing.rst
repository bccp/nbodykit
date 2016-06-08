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
- `kdcount`_   : pair-counting and friend-of-friend clustering with KD-Tree
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
~~~~~~~~~~~~~~~~~

- `pandas`_ and `pytables`_ are required to access the `pandas` subset of HDF5 and fast parsing of plain text files

.. _pandas: http://pandas.pydata.org/
.. _pytables: http://www.pytables.org

For creating data using a Halo Occupation Distribution
~~~~~~~~~~~~~~~~~~~~~~~

- `halotools`_ provides a general framework for populating halos with galaxies using Halo Occupation Distribution modeling and is required to access the `DataSource` that provides this feature

.. _halotools: http://halotools.readthedocs.io/en/latest/

Build Instructions
------------------

The software is designed to be installed with the ``pip`` utility like a regular
Python package. Using nbodykit from the source tree is not supported. See `Development Mode`_ for details.

The steps listed here are intended for a commodity Linux based cluster 
(e.g., a _`Rocks Cluster`) or a Linux based workstation / laptop.
Please note that there are slight changes to the procedure on systems running
a Mac OS X operating system and for Cray super-computers,
as explictly noted below in `Special notes for Mac and Cray`_.

Install the main nbodykit package, as well as the external dependencies 
listed above, into the default Python installation directory with:

.. code:: sh
   
    git clone http://github.com/bccp/nbodykit
    cd nbodykit

    # It may take a while to build fftw and pfft.
    # Mac and Edison are special, see notes below

    pip install -r requirements.txt
    pip install -U --force --no-deps .

A different installation directory can be specified via the ``--user`` or ``--root <dir>`` 
options of the ``pip install`` command. 

The pure-Python nbodykit package (without external dependencies) can be installed by 
omitting the ``-r requirements.txt`` option, with such an installation only requiring ``numpy``. 
Under such circumstances, the functionality of the package is greatly diminished -- package behavior 
in this instance is not tested and considered undefined. 

The dependencies of nbodykit are not fully stable, thus we recommend updating
the external dependencies occassionally via the ``-U`` option of ``pip install``. 
Also, since nbodykit is not yet stable enough for versioned releases, 
``--force`` ensures the current sourced version is installed:

.. code:: sh

    pip install -U -r requirements.txt
    pip install -U --force --no-deps .

To confirm that nbodykit is working, we can type, in a interactive Python session:

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
    mpirun -n 4 python-mpi nbkit.py -h

.. _`Rocks Cluster`: http://rocksclusters.org

Development Mode
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~

The ``autotools`` software is needed on a Mac:

.. code::

    sudo port install autoconf automake libtool
    
Additionally, on a Mac, the ``LDSHARED`` environment variable must be explicitly set. In bash, the installation command is

.. code::

    export LDSHARED="mpicc -bundle -undefined dynamic_lookup -DOMPI_IMPORTS"; pip install -r requirements.txt .
    
Using recent versions of MacPorts, we also need to tell ``mpicc`` to use ``gcc`` rather than the default ``clang``
compiler, which doesn't compile ``fftw`` correctly due to the lack of ``openmp`` support:

.. code::
    
    export OMPI_CC=gcc
 
NERSC Notes
~~~~~~~~~~~

We maintain a daily build of nbodykit on `NERSC`_ systems that works with the ``2.7-anaconda`` Python module 
and uses the `python-mpi-bcast`_ helper tool for fast startup of Python. Please see `this tutorial`_
for further details about using ``python-mpi-bcast`` for launching of Python applications on NERSC.

Below is an example job script, that runs the friends-of-friends finder (FOF) algorithm and then measures 
the power spectrum for halos with masses greater than :math:`10^{13} h^{-1} M_\odot`

.. code:: bash

    #! /bin/bash

    #SBATCH -n 32
    #SBATCH -p debug
    #SBATCH -t 10:00

    set -x
    module load python/2.7-anaconda

    source /project/projectdirs/m779/nbodykit/activate.sh

    srun -n 16 python-mpi $NBKITBIN FOF <<EOF
    nmin: 10
    datasource:
        plugin: FastPM
        path: data/fastpm_1.0000
    linklength: 0.2 
    output: output/fof_ll0.200_1.0000.hdf5
    calculate_initial_position: True
    EOF

    srun -n 16 python-mpi $NBKITBIN FFTPower <<EOF

    mode: 2d      # 1d or 2d
    Nmesh: 256     # size of mesh

    # here are the input fields 
    field:
      DataSource:
        plugin: FOFGroups
        path: output/fof_ll0.200_1.0000.hdf5
        m0: 2.27e12  # mass of a particle: use OmegaM * 27.75e10 * BoxSize ** 3/ Ntot
        rsd: z       # direction of RSD, usually use 'z', the default LOS direction.
    #    select: Rank < 1000 # Limits to the first 1000 halos for abundance matching or
        select: LogMass > 13 # limit to the halos with logMass > 13 (LRGs). 
    output: output/power_2d_fofgroups.dat  # output
    EOF

Note that we need to provide the mass of a single particle for the :class:`FOFGroups` DataSource. The number is :math:`27.75\times10^{10} \Omega_M (L / N)^3 h^{-1} M_\odot`, where :math:`L` is the boxsize in Mpc/h, and 
:math:`N` is the number of particles per side, such that :math:`N^3` is the total number of particles. 

.. _`NERSC`: http://www.nersc.gov/systems/
.. _`this tutorial`: https://github.com/rainwoodman/python-mpi-bcast/wiki/NERSC

Developing on NERSC
-------------------
If you would like to develop the code directly on NERSC (not recommended for the typical user), more installation work is required.

To install nbodykit on a NERSC Cray system (e.g. `Edison`_, `Cori`_), we need to ensure the Python environment
is set up to work efficiently on the computing nodes.

If `darshan`_ or `altd`_ are loaded by default, be sure to unload them before installing, as they tend to interfere
with Python:

.. code::

    module unload darshan
    module unload altd

and preferentially, use GNU compilers from PrgEnv-gnu

.. code::

    module unload PrgEnv-intel
    module unload PrgEnv-cray
    module load PrgEnv-gnu

then load the `Anaconda`_ Python distribution,

.. code::

    module load python/2.7-anaconda

We will need to set up fast Python start-up on a Cray computer, since
the default start-up scales badly with the number of processes. Our
preferred method is to use the `python-mpi-bcast`_ tool. 

1. Modify the shell profile, and set PYTHONUSERBASE to a unique location.
   (e.g. a path you have access on /project) for each machine.

   For example, this is excertion from the profile of 
   a typical user on NERSC (``.bash_profile.ext``),
   that has access to ``/project/projectdirs/m779/yfeng1``.

.. code:: bash

    if [ "$NERSC_HOST" == "edison" ]; then
        export PYTHONUSERBASE=/project/projectdirs/m779/yfeng1/local-edison
    fi

    if [ "$NERSC_HOST" == "cori" ]; then
        export PYTHONUSERBASE=/project/projectdirs/m779/yfeng1/local-cori
    fi

    export PATH=$PYTHONUSERBASE/bin:$PATH
    export LIBRARY_PATH=$PYTHONUSERBASE/lib
    export CPATH=$PYTHONUSERBASE/include

2. Install nbodykit to your user base with ``pip install --user``. 
   Also, create a bundle (tarball) of nbodykit. 
   Repeat this step if nbodykit (or any dependency) is updated.

.. code:: bash

    cd nbodykit;

    MPICC=cc pip install --user -r requirements $PWD

    # enable python-mpi-bcast (On NERSC)
    source /project/projectdirs/m779/python-mpi/activate.sh

    # create the bundle
    MPICC=cc bundle-pip nbodykit.tar.gz -r requirements.txt $PWD

.. _`Edison`: https://www.nersc.gov/users/computational-systems/edison/
.. _`Cori`: https://www.nersc.gov/users/computational-systems/cori
.. _`darshan`: http://www.mcs.anl.gov/research/projects/darshan/
.. _`altd`: http://www.nersc.gov/users/software/programming-libraries/altd/
.. _`Anaconda`: http://docs.continuum.io/anaconda/index
.. _`python-mpi-bcast`: https://github.com/rainwoodman/python-mpi-bcast