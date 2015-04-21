nbodykit
========

Software kit for N-body simulations. From particle mesh simulation to analysis.

Build
-----

The software is used in tree. First clone with

.. code :: sh
   
    git clone http://github.com/bccp/nbodykit
    cd nbodykit

Then build with

.. code :: sh

    ./build.sh

It may take a while to build fftw and pfft.

Packages are ready to use after importing the nbodykit namespace.

.. code :: python

    import nbodykit

    print(nbodykit)

Note that actual packages are still under their own namespaces, for example

.. code :: python

    import kdcount

    import pypm

This is to maintain the relative independence of the packages; but up to debate
may be changed.

Only two exceptions are packages specific to nbodykit.

.. code :: python

    from nbodykit import tpm
    from nbodykit import distributedarray

.. todo ::

    Some other code pieces in pypm shall be migrated to nbodykit as well. Most notouriously
    those doing QPM simulations under the nbody directory.

    qrpm is currently not integrated to the kit. It is quicker particle mesh mock code. The
    goal is to add an embeded python intepreter (currently has lua) to the root rank.


