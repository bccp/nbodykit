.. currentmodule:: nbodykit.base.catalog

Dealing with Discrete Data
==========================

The main interface for dealing with data in the form of catalogs of discrete
objects is provided by subclasses of the
:class:`nbodykit.base.catalog.CatalogSource` object.
In this section, we provide an overview of the class and note important
things to know.

.. contents::
   :depth: 2
   :local:
   :backlinks: none


What is a :class:`CatalogSource`?
---------------------------------

Most often the user starts with a catalog of discrete objects, with a set of
fields describing each object, such as the position coordinates, velocity,
mass, etc. Given this input data, the user wishes to use nbodykit to perform a
task, i.e., computing the power spectrum or grouping together objects with a
friends-of-friends algorithm. To achieve these goals, nbodykit provides the
:class:`nbodykit.base.catalog.CatalogSource` object.

The :class:`CatalogSource` object behaves much like a
`numpy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_,
where the fields of the array are referred to as "columns". These columns store
the information about the objects in the catalog; common columns are
"Position", "Velocity", "Mass", etc. A list of the column names
that are valid for a given catalog can be accessed via the
:attr:`CatalogSource.columns` attribute.

Use Cases
---------

The :class:`CatalogSource` is an abstract base class -- it cannot be directly
initialized. Instead, nbodykit includes several specialized catalog objects
in the :mod:`nbodykit.source.catalog` module. In general, these subclasses
fall into two categories:

#. Reading data from disk (see :ref:`reading-catalogs`)
#. Generating mock data at run time (see :ref:`mock-catalogs`)

Requirements
------------

A well-defined size
^^^^^^^^^^^^^^^^^^^

The only requirement to initialize a :class:`CatalogSource` is that the object
has a well-defined size. Information about the length of a :class:`CatalogSource`
is stored in two attributes:

- :attr:`CatalogSource.size` : the **local** size of the catalog, equal to the
  number of objects in the catalog on the local rank
- :attr:`CatalogSource.csize` : the **collective**, global size of the catalog,
  equal to the sum of :attr:`~CatalogSource.size` across all MPI ranks

So, the user can think of a :class:`CatalogSource` object has storing
information for a total of :attr:`~CatalogSource.csize` objects, which is
divided amongst the available MPI ranks such that each process only stores
information about :attr:`~CatalogSource.size` objects.

The ``Position`` column
^^^^^^^^^^^^^^^^^^^^^^^

All :class:`CatalogSource` objects must include the ``Position`` column, which
should be a ``(N,3)`` array giving the Cartesian position of each of the ``N``
objects in the catalog.

Often, the user will have the Cartesian coordinates
stored as separate columns or have the object positions in terms of
right ascension, declination, and redshift coordinates. See :ref:`common-operations`
for more details about how to construct the ``Position`` column for
these cases.

Default Columns
---------------

By default, all :class:`CatalogSource` objects include two default columns:

* ``Weight``
  The value to use for each particle when interpolating a :class:`CatalogSource`
  on to a mesh. By default, this array is set to unity for all objects.

* ``Selection``
  A boolean column that selects a subset slice of the :class:`CatalogSource`.
  By default, this column is set to ``True`` for all objects.

Storing Meta-data
-----------------

For all :class:`CatalogSource` objects, the input parameters and additional
meta-data are stored in the :attr:`CatalogSource.attrs` dictionary attribute.

API
---

For more information about specific catalog objects, please see the
:ref:`API section<api-discrete-data>`.
