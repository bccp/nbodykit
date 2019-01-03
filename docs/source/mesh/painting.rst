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

The :func:`~nbodykit.base.mesh.MeshSource.compute` function paints mass-weighted
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

Operations
----------

The painted field is an instance of :class:`pmesh.pm.RealField`. Methods
are provided for Fourier transforms (to a :class:`pmesh.pm.ComplexField` object),
and transfer functions on both the Real and Complex fields:

.. code:: python

   field = mesh.compute()

   def tf(k, v):
      return 1j * k[2] / k.normp(zeromode=1) ** 0.5 * v

   gradz = field.apply(tf).c2r()

The underlying numerical values of the field can be accessed via indexing.
A RealField is distributed across the entire MPI communicator of the mesh object,
and in general each single rank in the MPI communicator only sees a region of the
field.

- numpy methods (e.g. `field[...].std()`
  that operates on the local field values only compute the results on
  a single rank, thus only correct when a single rank is used:

- collective methods provide the correct result that has been reduced
  on the entire MPI communicator. For example, to compute the standard
  deviation of the field in a script that runs on sevearl MPI ranks,
  we shall use :code:`((field ** 2).cmean() - field.cmean() ** 2) ** 0.5` instead
  of :code:`field[...].std()`.

The positions of the grid points on which the field value resides
can be obtained from

.. code:: python

   field = mesh.compute()

   grid = field.pm.generate_uniform_particle_grid(shift=0)

A low resolution projected preview of the field can be obtained
(the example is along x-y plain)

.. code:: python

   field = mesh.compute()

   imshow(field.preview(Nmesh=64, axes=[0, 1]).T,
          origin='lower',
          extent=(0, field.BoxSize[0], 0, field.BoxSize[1]))

Shot-noise
----------

The shot-noise level of a weighted field is given by

.. math::

    SN = L^3 \frac{\sum W^2}{(\sum W)^2}

where `L^3` is the total volume of the box, and `W` is the weight of individual objects.
We see in the limit where `W=1` everywhere, the shotnoise is simply :math:`1 / \bar{n}`.

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
the :ref:`/cookbook/painting.ipynb#Painting-the-Line-of-sight-Momentum-Field`
recipe for further details.

Another common example is setting the weights :math:`W(\vx)` to a
column representing mass and painting multiple species of particles
to the same mesh using the
:class:`~nbodykit.source.catalog.species.MultipleSpeciesCatalog` object.
See the
:ref:`/cookbook/painting.ipynb#Painting-Multiple-Species-of-Particles`
recipe for more details.
