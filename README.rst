nbodykit
========

Software kit for N-body simulations. From particle mesh simulation to analysis.

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

It may take a while to build fftw and pfft.

.. note::

    On Edison, remember to unload darshan

        module unload darshan

    and preferentially, load PrgEnv-gnu

        module unload PrgEnv-intel
        module unload PrgEnv-gray
        module load PrgEnv-gnu

    then load python

        module load python
        module load cython
        module load numpy
        module load mpi4py


Packages are ready to use after importing the nbodykit namespace.

.. code:: python

    import nbodykit

    print(nbodykit)

Note that actual packages are still under their own namespaces, for example

.. code:: python

    import kdcount

    import pypm

This is to maintain the relative independence of the packages; but up to debate
may be changed.

Only two exceptions are packages specific to nbodykit.

.. code:: python

    from nbodykit import tpm
    from nbodykit import distributedarray


Some other code pieces in pypm shall be migrated to nbodykit as well. Most notouriously
those doing QPM simulations under the nbody directory.

qrpm is currently not integrated to the kit. It is quicker particle mesh mock code. The
goal is to add an embeded python intepreter (currently has lua) to the root rank.

.. _`pfft-python`: http://github.com/rainwoodman/pfft-python
.. _`pfft`: http://github.com/mpip/pfft
.. _`pypm`: http://github.com/rainwoodman/pypm
.. _`kdcount`: http://github.com/rainwoodman/kdcount
.. _`sharedmem`: http://github.com/rainwoodman/sharedmem
.. _`MP-sort`: http://github.com/rainwoodman/MP-sort
.. _`qrpm`: http://github.com/rainwoodman/qrpm
