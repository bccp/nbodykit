|
|

.. image:: _static/nbodykit-logo.gif
   :width: 425 px
   :align: center

|
|

a massively parallel, large-scale structure toolkit
===================================================

**nbodykit** is a massively parallel, open source project and Python
package providing a set of state-of-the-art algorithms useful in the analysis
of cosmological datasets from N-body simulations and large-scale structure surveys.

Driven by the optimism regarding the abundance and availability of
large-scale computing resources in the future, the development of nbodykit
distinguishes itself from other similar software packages
(i.e., `nbodyshop`_, `pynbody`_, `yt`_, `xi`_) by focusing on:

- a **unified** treatment of simulation and observational datasets by
  insulating algorithms from data containers

- reducing wall-clock time by **scaling** to thousands of cores

- **deployment** and availability on large, super computing facilities

All algorithms are parallel and run with Message Passing Interface (MPI).

.. _nbodyshop: http://www-hpcc.astro.washington.edu/tools/tools.html
.. _pynbody: https://github.com/pynbody/pynbody
.. _yt: http://yt-project.org/
.. _xi: http://github.com/bareid/xi

.. _getting-started:

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install.rst
   intro.rst
   quickstart/index.rst
   cookbook/index.rst

.. _playing-with-data:

.. toctree::
  :maxdepth: 1
  :caption: Playing with Data

  data/reading.rst
  data/on-demand-io.rst
  data/common-operations.rst
  data/painting.rst
  data/generating.rst

.. _getting-results:

.. toctree::
  :maxdepth: 1
  :caption: Getting Results

  algorithms/index.rst
  batch-mode.rst
  analyzing-results.rst

.. _help:

.. toctree::
  :maxdepth: 1
  :caption: Help and Reference

  api/index.rst
  development-guide.rst
  contact-support.rst
  changelog.rst
