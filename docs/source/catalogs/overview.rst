.. currentmodule:: nbodykit.base.catalog

Dealing with Discrete Data
==========================

The main interface for dealing with data in the form of catalogs of discrete
objects is provided by subclasses of the
:class:`nbodykit.base.catalog.CatalogSource` object.
In this section, we provide an overview of this class and note important
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
:class:`nbodykit.base.catalog.CatalogSource` base class.

The :class:`CatalogSource` object behaves much like a
:doc:`numpy structured array <numpy:user/basics.rec>`,
where the fields of the array are referred to as "columns". These columns store
the information about the objects in the catalog; common columns are
"Position", "Velocity", "Mass", etc. A list of the column names
that are valid for a given catalog can be accessed via the
:attr:`CatalogSource.columns` attribute.

Use Cases
---------

The :class:`CatalogSource` is an abstract base class -- it cannot be directly
initialized. Instead, nbodykit includes several specialized catalog subclasses
of :class:`CatalogSource` in the :mod:`nbodykit.source.catalog` module. In
general, these subclasses fall into two categories:

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

So, the user can think of a :class:`CatalogSource` object as storing
information for a total of :attr:`~CatalogSource.csize` objects, which is
divided amongst the available MPI ranks such that each process only stores
information about :attr:`~CatalogSource.size` objects.

The ``Position`` column
^^^^^^^^^^^^^^^^^^^^^^^

All :class:`CatalogSource` objects must include the ``Position`` column, which
should be a ``(N,3)`` array giving the Cartesian position of each of the ``N``
objects in the catalog.

Often, the user will have the Cartesian coordinates
stored as separate columns or have the object coordinates in terms of
right ascension, declination, and redshift. See :ref:`common-operations`
for more details about how to construct the ``Position`` column for
these cases.

.. _catalog-source-default-columns:

Default Columns
---------------

All :class:`CatalogSource` objects include several default columns.
These columns are used broadly throughout nbodykit and can be summarized as
follows:

============= ======================= ==================
**Name**      **Description**         **Default Value**
``Weight``    |Weight-Description|    1.0
``Value``     |Value-Description|     1.0
``Selection`` |Selection-Description| ``True``
============= ======================= ==================

.. |Weight-Description| replace::
  The weight to use for each particle when interpolating a :class:`CatalogSource`
  on to a mesh. The mesh field is a weighted average of ``Value``, with the weights
  given by ``Weight``.

.. |Value-Description| replace::
  When interpolating a :class:`CatalogSource` on to a mesh, the value of this
  array is used as the field value that each particle contributes to a given
  mesh cell. The mesh field is a weighted average of ``Value``, with the weights
  given by ``Weight``. For example, the ``Value`` column could represent
  ``Velocity``, in which case the field painted to the mesh will be momentum
  (mass-weighted velocity).

.. |Selection-Description| replace::
  A boolean column that selects a subset slice of the :class:`CatalogSource`.
  When converting a :class:`CatalogSource` to a mesh object, only the objects
  where the ``Selection`` column is ``True`` will be painted to the mesh.

Storing Meta-data
-----------------

For all :class:`CatalogSource` objects, the input parameters and additional
meta-data are stored in the :attr:`~CatalogSource.attrs` dictionary attribute.

API
---

For more information about specific catalog objects, please see the
:ref:`API section<api-discrete-data>`.
