.. currentmodule:: nbodykit.base.catalog

.. _on-demand-io:

On Demand IO via :mod:`dask.array`
==================================

nbodykit uses the `dask <http://dask.pydata.org>`_ package to store the columns
in :class:`CatalogSource` objects. The :mod:`dask` package implements
a :class:`dask.array.Array` object that mimics that interface of the more
familiar numpy array. In this section, we describe what exactly a dask array
is and how it is used in nbodykit.

.. _what-is-dask-array:

What is a dask array?
---------------------

In nbodykit, the dask array object is a data container that behaves nearly
identical to a numpy array, except for one key difference. When performing
manipulations on a numpy array, the operations are performed immediately.
This is not the case for dask arrays. Instead, dask arrays store these
operations in a task graph and only evaluate the operations when the user
specifies so via a call to a :func:`compute` function. Often the first
task in this graph is loading the data from disk, so dask provides
on-demand IO, allowing the user to control when data is read from disk.

It is useful to describe a bit more about the nuts and bolts of the dask
array to illustrate its full power. The dask array object
cuts up the full array into many smaller arrays and performs calculations
on these smaller "chunks". This allows array computations to be
performed on large data that does not fit into memory
(but can be stored on disk). Particularly useful on laptops and other
systems with limited memory, it extends the maximum size of useable datasets
from the size of memory to the size of the disk storage.

By Example
----------

The dask array functionality is best illustrated by example. Here, we
initialize a :class:`~nbodykit.source.catalog.uniform.UniformCatalog`
that generates objects with uniformly distributed position and velocity columns.

.. ipython:: python

  from nbodykit.lab import UniformCatalog

  cat = UniformCatalog(nbar=100, BoxSize=1.0, seed=42)

  print(cat)
  print(cat['Position'])

We see that the ``Position`` column can be accessed by indexing the catalog
with the column name and that the returned object is not a numpy array
but a dask array. The dask array has the same ``shape`` (96,3) and
``dtype`` ('f8') as the underlying numpy array but also includes the
``chunksize`` attribute. This attribute specifies the size of the internal
chunks that dask uses to examine arrays in smaller pieces. In this case,
the data size is small enough that only a single chunk is needed.

.. _dask-array-module:

The :mod:`dask.array` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`dask.array` module provides much of the same functionality as the
:mod:`numpy` module, but with functions optimized to perform operations on
dask arrays.

For example, we can easily compute the minimum and maximum position coordinates
using the :func:`dask.array.min` and :func:`dask.array.max` functions.

.. ipython:: python

    import dask.array as da

    pos = cat['Position']
    minpos = da.min(pos, axis=0)
    maxpos = da.max(pos, axis=0)

    print("minimum position coordinates = ", minpos)
    print("maximum position coordinates = ", maxpos)

Here, we see that the result of our calls to :func:`dask.array.min` and
:func:`dask.array.max` are also stored as dask arrays. The task
has not yet been performed but instead added to the internal dask task graph.

For a full list of the available functionality, please see the
`dask array documentation <http://dask.pydata.org/en/latest/array-api.html>`_.
Most of the most commonly used functions in numpy have implementations in the
:mod:`dask.array` module. In addition to these functions, dask arrays support
the usual array arithmetic operations. For example, to re-scale the
position coordinate array:

.. ipython:: python

    BoxSize = 2500.0
    pos *= BoxSize

    rescaled_minpos = da.min(pos, axis=0)
    rescaled_maxpos = da.max(pos, axis=0)


.. _evaluating-dask-array:

Evaluating a dask array
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`CatalogSource.compute` function computes a dask array and returns
the result of the internal series of tasks, either a numpy array or float.
For example, we can compute the minimum and maximum of the position coordinates
using:

.. ipython:: python

    minpos, maxpos = cat.compute(minpos, maxpos)
    print("minimum position coordinates = ", minpos)
    print("maximum position coordinates = ", maxpos)

And similarly, we see the result of the rescaling operation earlier:


.. ipython:: python

    minpos, maxpos = cat.compute(rescaled_minpos, rescaled_maxpos)
    print("minimum re-scaled position coordinates = ", minpos)
    print("maximum re-scaled position coordinates = ", maxpos)

.. _caching-with-dask:

Caching with Dask
-----------------

Subclasses of :class:`CatalogSource` accept the ``use_cache`` keyword, which
can turn on an internal cache to use when evaluating dask arrays. Often
the most expensive task of evaluating a dask array is loading the data
from disk. With this feature turned on, the :class:`CatalogSource` object will
cache intermediate results, such that repeated calls to
:func:`CatalogSource.compute` do not repeat expensive IO operations.

.. note::

    All dask arrays also have a built-in :func:`compute` function that can be
    called to evaluate the array. However, this function does not take advantage
    of any cache features. The built-in :func:`compute` function is useful
    for quick data inspection, but we recommend using
    :func:`CatalogSource.compute` when performing most calculations with nbodykit.

.. _larger-than-memory-arrays:

Examining Larger-than-Memory Data
---------------------------------

:class:`CatalogSource` objects automatically take advantage of the chunking
features of the dask array, greatly reducing the difficulties of
analyzing larger-than-memory data. When combined with the ability of the
:class:`CatalogSource` object to provide a continuous view of multiple files
at once, we can analyze large amounts of data from a single catalog with ease.

A common use case is examining a directory of large binary outputs from a
N-body simulation on a laptop. Often the user wishes to select a smaller subsample
of the catalog or perform some trivial data inspection to verify that accuracy of
the data. These tasks become straightforward with nbodykit, using the functionality
provided by the :class:`CatalogSource` object and the :mod:`dask` package.
The nbodykit :class:`CatalogSource` interface remains the same, regardless
of the size of the data that the user is loading.
