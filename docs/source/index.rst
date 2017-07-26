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
package providing a set of state-of-the-art, large-scale structure algorithms
useful in the analysis of cosmological datasets from N-body simulations and
observational surveys.

Driven by the optimism regarding the abundance and availability of
large-scale computing resources in the future, the development of nbodykit
distinguishes itself from other similar software packages
(i.e., `nbodyshop`_, `pynbody`_, `yt`_, `xi`_) by focusing on:

- a **unified** treatment of simulation and observational datasets by
  insulating algorithms from data containers

- support for a wide **variety of data** formats, as well as **large volumes of data**

- reducing wall-clock time by **scaling** to thousands of cores

- **deployment** and availability on large, super-computing facilities

- an **interactive** user interface that performs as well in a Jupyter
  notebook as on super-computing machines

All algorithms are parallel and run with Message Passing Interface (MPI).
For a list of the algorithms currently implemented, see :ref:`available-algorithms`.

The source code is publicly available at https://github.com/bccp/nbodykit.

.. _nbodyshop: http://www-hpcc.astro.washington.edu/tools/tools.html
.. _pynbody: https://github.com/pynbody/pynbody
.. _yt: http://yt-project.org/
.. _xi: http://github.com/bareid/xi

.. _getting-started:

Getting Started
---------------

* :doc:`install`
* :doc:`intro`
* :doc:`quickstart/index`
* :doc:`cookbook/index`

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   install
   intro
   quickstart/index
   cookbook/index

.. _playing-with-data:

Discrete Data Catalogs
----------------------

* :doc:`Overview <catalogs/overview>`
* :doc:`catalogs/reading`
* :doc:`catalogs/on-demand-io`
* :doc:`catalogs/common-operations`

.. toctree::
  :maxdepth: 1
  :caption: Discrete Data Catalogs
  :hidden:

  Overview <catalogs/overview>
  catalogs/reading
  catalogs/generating
  catalogs/on-demand-io
  catalogs/common-operations

Data on a Mesh
--------------

* :doc:`Overview <mesh/overview>`
* :doc:`mesh/painting`
* :doc:`mesh/reading`
* :doc:`mesh/generating`

.. toctree::
  :maxdepth: 1
  :caption: Data on a Mesh
  :hidden:

  Overview <mesh/overview>
  mesh/painting
  mesh/reading
  mesh/generating

.. _getting-results:

Getting Results
---------------

* :doc:`algorithms/index`
* :doc:`batch-mode`
* :doc:`analyzing-results`

.. toctree::
  :maxdepth: 1
  :caption: Getting Results
  :hidden:

  algorithms/index
  batch-mode
  analyzing-results

.. _help:

Help and Reference
------------------

* :doc:`api/api`
* :doc:`development-guide`
* :doc:`contact-support`
* :doc:`changelog`

.. toctree::
  :maxdepth: 1
  :caption: Help and Reference
  :hidden:

  api/api
  development-guide
  contact-support
  changelog
