|
|

.. image:: _static/nbodykit-logo.gif
   :width: 425 px
   :align: center

|
|

.. title:: nbodykit documentation

a massively parallel, large-scale structure toolkit
===================================================

**nbodykit** is an open source project written in Python
that provides a set of state-of-the-art, large-scale structure algorithms
useful in the analysis of cosmological datasets from N-body simulations and
observational surveys. All algorithms are massively parallel and run using the
Message Passing Interface (MPI).

Driven by the optimism regarding the abundance and availability of
large-scale computing resources in the future, the development of nbodykit
distinguishes itself from other similar software packages
(i.e., `nbodyshop`_, `pynbody`_, `yt`_, `xi`_) by focusing on:

- a **unified** treatment of simulation and observational datasets by
  insulating algorithms from data containers

- support for a wide **variety of data** formats, as well as **large volumes of data**

- the ability to reduce wall-clock time by **scaling** to thousands of cores

- **deployment** and availability on large, supercomputing facilities

- an **interactive** user interface that performs as well in a `Jupyter
  notebook`_ as on a supercomputing machine

.. _nbodyshop: http://www-hpcc.astro.washington.edu/tools/tools.html
.. _pynbody: https://github.com/pynbody/pynbody
.. _yt: http://yt-project.org/
.. _xi: http://github.com/bareid/xi
.. _Jupyter notebook: http://jupyter-notebook.rtfd.io

----

Learning By Example
-------------------

For users who wish to dive right in, an interactive environment
containing :ref:`our cookbook recipes <cookbook>` is available to users
via the `BinderHub service <https://github.com/jupyterhub/binderhub>`_.
Just click the launch button below to get started!

.. figure:: http://mybinder.org/badge.svg
    :alt: binder
    :target: https://mybinder.org/v2/gh/bccp/nbodykit-cookbook/master?filepath=recipes

See :ref:`cookbook` for descriptions of the various notebook recipes.

----

Getting nbodykit
----------------

To get up and running with your own copy of nbodykit, please follow the
:doc:`installation instructions <getting-started/install>`.
nbodykit is currently supported on macOS and Linux architectures. The
recommended installation method uses the
`Anaconda <https://www.continuum.io/downloads>`_ Python distribution.

nbodykit is compatible with **Python versions 2.7, 3.5, and 3.6**, and the
source code is publicly available at https://github.com/bccp/nbodykit.

----

.. _getting-started:

A 1 minute introduction to nbodykit
-----------------------------------

To start, we initialize the nbodykit ":mod:`~nbodykit.lab`":

.. code:: python

    from nbodykit.lab import *

There are two core data structures in nbodykit: catalogs and meshes. These
represent the two main ways astronomers interact with data in large-scale
structure analysis. Catalogs hold information describing a set of discrete
objects, storing the data in columns. nbodykit includes functionality for
initializing catalogs from a variety of file formats as well as more advanced
techniques for generating catalogs of simulated particles.

Below, we create a very simple catalog of uniformly distributed particles in a
box of side length :math:`L = 1 \ h^{-1} \mathrm{Mpc}`:

.. code:: python

    catalog = UniformCatalog(nbar=100, BoxSize=1.0)

Catalogs have a fixed size and a set of columns describing the particle data.
In this case, our catalog has "Position" and "Velocity" columns. Users can
easily manipulate the existing column data or add new columns:

.. code:: python

    BoxSize = 2500.
    catalog['Position'] *= BoxSize # re-normalize units of Position
    catalog['Mass'] = 10**(numpy.random(12, 15, size=len(catalog))) # add some random mass values

We can generate a representation of the density field on a mesh using our
catalog of objects. Here, we interpolate the particles onto a mesh of
size :math:`64^3`:

.. code:: python

    mesh = catalog.to_mesh(Nmesh=64, BoxSize=BoxSize)

We can save our mesh to disk to later re-load using nbodykit:

.. code:: python

    mesh.save('mesh.bigfile')

or preview a low-resolution, 2D projection of the mesh to make sure everythings
looks as expected:

.. code:: python

    import matplotlib.pyplot as plt
    plt.imshow(mesh.preview(axes=[0,1], Nmesh=32))

Finally, we can feed our density field mesh in to one of the nbodykit algorithms.
For example, below we use the :class:`~nbodykit.algorithms.fftpower.FFTPower` algorithm to
compute the power spectrum :math:`P(k,\mu)` of the density
mesh using a fast Fourier transform via

.. code:: python

    result = FFTPower(mesh, Nmu=5)

with the measured power stored as the ``power`` attribute of the
``result`` variable. The algorithm result and meta-data, input
parameters, etc. can then be saved to disk as a JSON file:

.. code:: python

    result.save("power-result.json")

It is important to remember that nbodykit is fully parallelized using MPI. This
means that the above code snippets can be excuted in a Jupyter notebook with
only a single CPU or using a standalone Python script with
an arbitrary number of MPI workers. We aim to hide as much of the parallel
abstraction from users as possible. When executing in parallel, data will
automatically be divided amongst the available MPI workers, and each worker
computes its own smaller portion of the algorithm result before finally
these calculations are combined into the final result.

----

Getting Started
---------------

We also provide detailed overviews of the two main data containers in
nbodykit, catalogs and meshes, and we walk through the necessary background
information for each of the available algorithms in nbodykit. The main
areas of the documentation can be broken down into the following sub-sections:

* :doc:`Introduction <getting-started/intro>`: an introduction to key nbodykit concepts and things to know
* :ref:`cosmology`: a guide to the cosmology-related functionality in nbodykit
* :ref:`intro-catalog-data`: a guide to dealing with catalogs of discrete data catalogs
* :ref:`intro-mesh-data`: an overview of data on a discrete mesh
* :ref:`getting-results`: an introduction to the available algorithms, parallel computation, and saving/analyzing results

.. _help:

----

Getting Help
------------

* :doc:`api/api`
* :doc:`help/support`
* :doc:`help/contributing`
* :doc:`help/changelog`

.. --------------------------------------------
.. include hidden toc tree for site navigation
.. -------------------------------------------

.. toctree::
  :maxdepth: 1
  :caption: Getting Started
  :hidden:

  getting-started/install
  getting-started/intro
  getting-started/cosmology.ipynb
  cookbook/index

.. toctree::
  :maxdepth: 1
  :caption: Discrete Data Catalogs
  :hidden:

  Overview <catalogs/overview>
  catalogs/reading.ipynb
  catalogs/on-demand-io.ipynb
  catalogs/common-operations.ipynb
  catalogs/mock-data.ipynb

.. toctree::
  :maxdepth: 1
  :caption: Data on a Mesh
  :hidden:

  Overview <mesh/overview.ipynb>
  mesh/creating.ipynb
  mesh/painting
  mesh/common-operations.ipynb

.. toctree::
  :maxdepth: 1
  :caption: Getting Results
  :hidden:

  results/algorithms/index
  Parallel Computation <results/parallel>
  results/analyzing
  results/saving

.. toctree::
  :maxdepth: 1
  :caption: Help and Reference
  :hidden:
  :includehidden:

  api/api
  help/support
  help/contributing
  help/changelog
