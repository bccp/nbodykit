nbodykit: a massively parallel large-scale structure toolkit
============================================================

**nbodykit** is an open source project and Python package providing 
a set of algorithms useful in the analysis of cosmological 
datasets from N-body simulations and large-scale structure surveys.

Driven by the optimism regarding the abundance and availability of 
large-scale computing resources in the future, the development of nbodykit
distinguishes itself from other similar software packages
(i.e., `nbodyshop`_, `pynbody`_, `yt`_, `xi`_) by focusing on :

- a **unified** treatment of simulation and observational datasets by 
  insulating algorithms from data containers

- reducing wall-clock time by **scaling** to thousands of cores

- **deployment** and availability on large, super computing facilities

All algorithms are parallel and run with Message Passing Interface (MPI). 

For users using the `NERSC`_ super-computers, we provide a ready-to-use tarball 
of nbodykit and its dependencies; see `Using nbodykit on NERSC <http://nbodykit.readthedocs.io/en/latest/installing.html#using-nbodykit-on-nersc>`_ for more details.

.. _nbodyshop: http://www-hpcc.astro.washington.edu/tools/tools.html
.. _pynbody: https://github.com/pynbody/pynbody
.. _yt: http://yt-project.org/
.. _xi: http://github.com/bareid/xi
.. _`NERSC`: http://www.nersc.gov/systems/

Build Status
------------

We perform integrated tests of the code, including all built-in algorithms, in a
miniconda environment for Python 2.7 and Python 3.4. 

.. image:: https://travis-ci.org/bccp/nbodykit.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/bccp/nbodykit
.. image:: https://coveralls.io/repos/github/bccp/nbodykit/badge.svg?branch=master 
    :alt: Test Coverage
    :target: https://coveralls.io/github/bccp/nbodykit?branch=master
.. image:: https://img.shields.io/pypi/v/nbodykit.svg
   :alt: PyPi
   :target: https://pypi.python.org/pypi/nbodykit/

Installation
------------

Refer to .travis.yml, the build section:

- https://github.com/bccp/nbodykit/blob/master/.travis.yml

Documentation
-------------

The official documentation is hosted on ReadTheDocs at http://nbodykit.readthedocs.org/. 
