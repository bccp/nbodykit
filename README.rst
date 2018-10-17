nbodykit: a massively parallel large-scale structure toolkit
============================================================

.. image:: https://zenodo.org/badge/34348490.svg
   :target: https://zenodo.org/badge/latestdoi/34348490

|

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

- **deployment** and availability on large, super-computing facilities

- an **interactive** user interface that performs as well in a `Jupyter
  notebook <http://jupyter.org>`_ as on super-computing machines

All algorithms are parallel and run with Message Passing Interface (MPI).

.. _nbodyshop: http://www-hpcc.astro.washington.edu/tools/tools.html
.. _pynbody: https://github.com/pynbody/pynbody
.. _yt: http://yt-project.org/
.. _xi: http://github.com/bareid/xi
.. _`NERSC`: http://www.nersc.gov/systems/

Build Status
------------

We perform integrated tests of the code, including all built-in algorithms, in a
miniconda environment for Python 2.7, 3.5, and 3.6.

.. image:: https://travis-ci.org/bccp/nbodykit.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/bccp/nbodykit
.. image:: https://coveralls.io/repos/github/bccp/nbodykit/badge.svg?branch=master
    :alt: Test Coverage
    :target: https://coveralls.io/github/bccp/nbodykit?branch=master
.. image:: https://img.shields.io/conda/v/bccp/nbodykit.svg
   :alt: Conda
   :target: https://anaconda.org/bccp/nbodykit
.. image:: https://img.shields.io/pypi/v/nbodykit.svg
   :alt: PyPi
   :target: https://pypi.python.org/pypi/nbodykit/

Documentation
-------------

The official documentation is hosted on ReadTheDocs at http://nbodykit.readthedocs.org/.

Cookbook Recipes
----------------

Users can dive right into an interactive cookbook of example recipes using binder.
We've compiled a set of Jupyter notebooks to help users learn nbodykit by example — just click the launch button below to get started!

.. image:: http://mybinder.org/badge.svg
    :alt: binder
    :target: https://mybinder.org/v2/gh/bccp/nbodykit-cookbook/master?filepath=recipes

|

Users can also view a static version of the cookbook recipes
`in the documentation <http://nbodykit.rtfd.io/en/latest/cookbook/index.html>`_.

Installation
------------

We recommend using the Anaconda distribution of Python. To obtain the
dependencies and install a package on OSX or Linux, use

.. code-block:: bash

    $ conda install -c bccp nbodykit

We are considering support for Windows, but this depends on the status
of `mpi4py`.

Using nbodykit on NERSC
-----------------------

On the Cori and Edison machines at NERSC, we maintain a nightly conda build of
the latest stable release of nbodykit. See
`the documentation <http://nbodykit.readthedocs.io/en/latest/install.html#nbodykit-on-nersc>`_
for using nbodykit on NERSC for more details.

Bumping to a new version
------------------------

1. git pull - confirm that the master branch is up-to-date
2. Edit Changelog (CHANGES.rst) - Make sure to include all issues which have arisen since the last version. (git add ... -> git commit -m "Update Changelog" -> git push)
3. Edit version.py -> git push ("bump version to ...")
4. Go to https://travis-ci.org/bccp/nbodykit and make sure it merged without any problems.
5. Go to bccp/conda-channel-bccp repo and do "Restart build"
6. git tag 0.3.? -> git push --tags
7. bump to a development version (0.3.?dev0)
