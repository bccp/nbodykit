nbodykit
========

Software kit for cosmological N-body simulations. 

Top Level Executables
=====================

The tools are provide as top level executable scripts. 
All tools are MPI parallel.  We can deal with very large simulations.

- ``bin/power.py`` is a Power Spectrum calculator.

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

.. _`pfft-python`: http://github.com/rainwoodman/pfft-python
.. _`pfft`: http://github.com/mpip/pfft
.. _`pypm`: http://github.com/rainwoodman/pypm
.. _`kdcount`: http://github.com/rainwoodman/kdcount
.. _`sharedmem`: http://github.com/rainwoodman/sharedmem
.. _`MP-sort`: http://github.com/rainwoodman/MP-sort
.. _`qrpm`: http://github.com/rainwoodman/qrpm

The software is built on top of existing tools. Please refer to their
documentations:

- `pfft`_    : massively parallel fast fourier transform, pencil domains
- `pfft-python`_  : python binding of pfft
- `pypm`_     :  particle mesh framework in Python
- `kdcount`_   : pair-counting and friend-of-friend clustering with KD-Tree
- `MP-sort`_   : massively parallel sorting 
- `sharedmem`_ : in-node parallelism with fork and copy-on-write.

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

    pip install -U -r requirements.txt -e .


Now we shall be able to use nbodykit, in a interactive python session 
(please remember to jump to bin/ directory to avoid weird issues about importing in-tree)

.. code:: python

    import nbodykit
    print(nbodykit)

    import kdcount
    print(kdcount)

    import pypm
    print(pypm)

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
 
**UPDATE BELOW THIS**  

Edison Notes
++++++++++++

On Edison, remember to unload darshan

.. code::

    module unload darshan

and preferentially, load PrgEnv-gnu

.. code::

    module unload PrgEnv-intel
    module unload PrgEnv-gray
    module load PrgEnv-gnu

then load python

.. code::

    module load python/2.7-anaconda

Also prefix the compiler MPICC=cc, so do this

.. code::
    
    MPICC=cc pip install -r requirements.txt -e .

Optionally, build the python-mpi-bcast bundle for massively parallel python jobs

.. code:: bash

    bash /project/projectdirs/m779/python-mpi/tar-anaconda.sh nbodykit-dependencies.tar.gz install/

We can also build a bundle that includes nbodykit:

.. code:: bash

    # in source code root

    python setup.py install --prefix=install
    bash /project/projectdirs/m779/python-mpi/tar-anaconda.sh nbodykit.tar.gz install/

