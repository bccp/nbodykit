nbodykit
========

Software kit for cosmological N-body simulations. 

Top Level Executables
=====================

The tools are provide as top level executable scripts. 
All tools are MPI parallel.  We can deal with very large simulations.

- power.py is a Power Spectrum calculator.

- fof.py is a friend of friend feature (halo) identifier.

- subhalo.py is a FOF6D sub-feature (subhalo) identifier .

Run them with '-h' to see the inline help.

Each takes a few 'datasource' strings that specify the format of the inputs.

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

The software is used in tree. First clone with

.. code:: sh
   
    git clone http://github.com/bccp/nbodykit
    cd nbodykit

Then build with

.. code:: sh

    ./build.sh

Note that sometimes 'git submodule update' is needed to sync the subpackages, do it often.

It may take a while to build fftw and pfft.

.. attention:: Mac Notes

    autotools are needed on a Mac
    
    .. code::
    
        sudo port install autoconf automake libtool
        
    On Mac, the `LDSHARED` environment variable must be explicitly set. In bash, the command is

    .. code::

        export LDSHARED="mpicc -bundle -undefined dynamic_lookup"; ./build.sh
        
   On recent versions of MacPorts, we also need to tell mpicc to use gcc rather than the default clang
   compiler, which doesn't compile fftw correctly due to lack of openmp support.
   
    .. code::
        
        export OMPI_CC=gcc
   
.. attention:: Edison Notes

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

        module load python
        module load cython
        module load numpy
        module load mpi4py
    
    Also prefix the compiler MPICC=cc

Packages are ready to use after importing the nbodykit namespace.

.. code:: python

    import nbodykit

    print(nbodykit)

Note that actual packages are still under their own namespaces, for example

.. code:: python

    import kdcount

    import pypm

This is to maintain the relative independence of the packages. 

