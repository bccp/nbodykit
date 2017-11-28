.. _painting-mesh:

.. currentmodule:: nbodykit.base.catalog

Painting Catalogs to a Mesh
===========================

The :func:`MeshSource.paint` function produces the values
of the field on the mesh, returning either a :class:`~pmesh.pm.RealField` or
:class:`~pmesh.pm.ComplexField`. In this section, we focus on the
process of interpolating a set of discrete objects in a :class:`CatalogSource`
on to a mesh and how users can customize this procedure.

The Painted Field
-----------------

The :func:`~nbodykit.base.mesh.MeshSource.paint` function paints mass-weighted
(or equivalently, number-weighted) fields to a mesh. So, when painting a
:class:`CatalogSource` to a mesh, the field :math:`F(\vx)` that is painted
is:

.. math::

    F(\vx) = \left[ 1 + \delta'(\vx) \right] V(\vx),

where :math:`V(\vx)` represents the field value painted to the mesh and
:math:`\delta'(\vx)` is the (weighted) overdensity field, given by:

.. math::

    \delta'(\vx) = \frac{n(\vx)'}{\bar{n}'} - 1,

where :math:`\bar{n}'` is the weighted mean number density of objects.
Here, quantities denoted with a prime (:math:`'`) indicate weighted quantities.
The unweighted number density field :math:`n(\vx)` is related to its
weighted counterpart via :math:`n'(\vx) = W(\vx) n(\vx)`, where
:math:`W(\vx)` are the weights.

Users can control the behavior of the value :math:`V(\vx)` and the weights
:math:`W(\vx)` when converting a :class:`CatalogSource` object to a mesh
via the :func:`~CatalogSource.to_mesh` function. Specifically, the
``weight`` and ``value`` keywords allow users to indicate the name of
the column in the :class:`CatalogSource` to use for :math:`W(\vx)`
and :math:`V(\vx)`. See :ref:`additional-mesh-config` for more details
on these keywords.

Default Behavior
----------------

The default behavior is :math:`W(\vx) = 1` and :math:`V(\vx) = 1`, in which
case the painted field is given by:

.. math::

    F^\mathrm{default}(\vx) = 1 + \delta(\vx).

In the :func:`CatalogSource.to_mesh` function, the default
values for the ``value`` and ``weight`` keywords are the ``Value`` and
``Weight`` columns, respectively. These are
:ref:`default columns <catalog-source-default-columns>` that are in all
:class:`CatalogSource` objects that are set to unity by default.

More Examples
-------------

The :ref:`cookbook-painting` section of the cookbook contains several more
examples that change the default behavior to paint customized fields to the
mesh.

For example, users can set :math:`V(\vx)` to a column holding a component
of the velocity field, in which case the painted field :math:`F(\vx)` would
represent the momentum (mass-weighted velocity) field. See
the :ref:`cookbook/painting.ipynb#Painting-the-Line-of-sight-Momentum-Field`
recipe for further details.

Another common example is setting the weights :math:`W(\vx)` to a
column representing mass and painting multiple species of particles
to the same mesh using the
:class:`~nbodykit.source.catalog.species.MultipleSpeciesCatalog` object.
See the
:ref:`cookbook/painting.ipynb#Painting-Multiple-Species-of-Particles`
recipe for more details.
