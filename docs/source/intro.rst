A Brief Introduction
====================

In this section, we provide a brief overview of the major functionality
of nbodykit, as well as an introduction to some of the technical jargon
needed to get up and running quickly. We try to familiarize the user with the
various aspects of nbodykit needed to take full advantage of nbodykit's
computing power. This section also serves as a nice outline of the documentation,
with links to more detailed descriptions included throughout.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The ``lab`` framework
---------------------

A core design goal of nbodykit is maintaining an interactive user
experience, allowing the user to quickly experiment and play around
with data sets and statistics, while still leveraging the power of
parallel processing when necessary. Motivated by the power of
`Jupyter notebooks <http://jupyter.org>`_, we adopt a ``lab``
framework for nbodykit, where all of the necessary data containers
and algorithms can be imported from a single module:

.. code-block:: python

  from nbodykit.lab import *

  [insert cool science here]

With all of the necessary tools now in hand, the user can easily load
a data set, compute statistics of that data via one of the
built-in algorithms, and save the results in just a few lines. The end
result is a reproducible scientific result, generated from clear
and concise code that flows from step to step.

Parallel Computation with MPI
-----------------------------

The nbodykit package is fully parallelized using the Python
bindings of the Message Passage Interface (MPI) available in ``mpi4py``. While
we aim to hide most of the complexities of MPI from the top-level user
interface, it is helpful to know some basic aspects of the MPI framework
for understanding how nbodykit works to compute its results. If you are
unfamiliar with MPI, a good place to start is the `documentation for
mpi4py <http://mpi4py.readthedocs.io/en/stable/intro.html>`_. Briefly,
MPI allows nbodykit to use a specified number of CPUs, which work independently
to achieve a common goal and pass messages back and forth to coordinate their
work.

.. note::

  It is important to keep in mind that memory is not shared across
  different CPUS when using MPI. This is particularly important when loading data
  in parallel using nbodykit, as the data is spread out evenly across all of the
  available CPUs. This allows nbodykit to load very large data sets quickly, given
  a necessary number of CPUs are available, which otherwise would not fit
  into the memory of a single CPU, given the prohibitively large size. However,
  a single CPU does not have access to the full dataset, but merely the portion
  stored in its memory (usually :math:`1/N` of the full data set, where
  :math:`N` is the number of CPUs).


Insulating Data from Algorithms
-------------------------------

A major goal of the project is to provide a unified treatment of
both simulation and observational datasets, allowing nbodykit to
be used in the analysis of not only N-body simulations, but also
data from current and future large-scale structure surveys.

Discrete Catalogs of Particles
------------------------------

From Catalogs to Mesh Data
--------------------------

Running your Favorite Algorithm
-------------------------------

Quickstart and Cookbook
-----------------------

Extending nbodykit
------------------
