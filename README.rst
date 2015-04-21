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

Packages are ready to use under nbodykit namespace.

.. code :: python

    import nbodykit

    print(nbodykit.pypm)

