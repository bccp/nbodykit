.. currentmodule:: nbodykit.base.mesh

Dealing with Data on a Mesh
===========================

.. contents::
   :depth: 2
   :local:
   :backlinks: none


What is a :class:`MeshSource`?
------------------------------

Often in large-scale structure data analysis, we wish to manipulate
representations of continuous quantities on a discrete grid. The canonical
example is the analysis of the cosmological density field,
interpolated on to a 3D mesh from a discrete set of galaxies. To support
such calculations, nbodykit provides the
:class:`nbodykit.base.mesh.MeshSource` object.

Fundamentally, the :class:`MeshSource` object stores a (possibly weighted)
density field on a three-dimensional mesh, with the ``Nmesh`` parameter
determining the number of grid cells per side (such that there are
:math:`\mathrm{Nmesh}^3` mesh cells).  nbodykit adds the functionality
to analyze these fields in both configuration space (often referred
to real space) and Fourier space through an interface to the
:class:`~pmesh.pm.RealField` and :class:`~pmesh.pm.ComplexField` objects
implemented by the :mod:`pmesh` package. These objects are
paired classes, related through the operation of a 3D
`fast Fourier transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_
(FFT). The FFT operation implemented in :mod:`pmesh` relies on
the `pfft-python <https://github.com/rainwoodman/pfft-python>`_ package, which
is a Python binding of `PFFT <github.com/mpip/pfft>`_, a massively parallel
FFT library.

Use Cases
---------

The :class:`MeshSource` is an abstract base class -- it cannot be directly
initialized. Instead, nbodykit includes several specialized subclasses of
:class:`MeshSource` in the :mod:`nbodykit.source.mesh` module. In general,
these subclasses fall into three categories:

#. Generating mesh data from a
   :class:`~nbodykit.base.catalog.CatalogSource` (see :ref:`catalog-to-mesh`)
#. Reading mesh data from disk (see :ref:`saving-loading-mesh`)
#. Generating mock fields directly on a mesh (see :ref:`gaussian-meshes`)

Painting the Mesh
-----------------

The :func:`MeshSource.paint` function produces the values of the field
on the mesh, returning either a :class:`~pmesh.pm.RealField` or
:class:`~pmesh.pm.ComplexField`. This function treats the mesh equally in
either configuration space or Fourier space, internally
performing the appropriate FFTs. By specifying the ``mode`` keyword to the
:func:`~MeshSource.paint` function, users can access either the field
data in configuration space or the complex modes of the field in Fourier space.

The "painting" nomenclature derives from the most common use case. The
process of interpolating a set of discrete objects on to the mesh evokes
the imagery of "painting" the mesh. More generally, the :func:`~MeshSource.paint`
function is responsible for filling in the mesh with data, which could also
involve reading data from disk or generating mock fields directly on the mesh.

For further details and examples of painting a catalog of discrete objects
to a mesh, see :ref:`painting-mesh`.

Fields: ``RealField`` and ``ComplexField``
------------------------------------------

The :class:`MeshSource` class provides an interface to the
:class:`pmesh.pm.RealField` and :class:`pmesh.pm.ComplexField` objects.
These classes behave like numpy arrays and include functions to
perform parallel forward and inverse FFTs. These
field objects are initialized from a :class:`pmesh.pm.ParticleMesh`, which
sets the number of mesh cells and stores FFT-related grid quantities.

.. ipython:: python

    from pmesh.pm import ParticleMesh, RealField, ComplexField

    # a 8^3 mesh
    pm = ParticleMesh(Nmesh=[8,8,8])

    # initialize a RealField
    rfield = RealField(pm)

    # shape
    print(rfield.shape)

    # set entire mesh to unity
    rfield[...] = 1.0

    # print the mean of the underlying array
    print(rfield.value.mean())

All :class:`MeshSource` objects implement either the
:func:`MeshSource.to_real_field` function or the
:func:`MeshSource.to_complex_field` function. These
functions are responsible for returning either a :class:`~pmesh.pm.RealField`
or a :class:`~pmesh.pm.ComplexField`. The :class:`MeshSource.paint` function
calls these functions, providing the core functionality
of the :class:`MeshSource` class.


The :func:`c2r` and :func:`r2c` functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can transform between :class:`~pmesh.pm.RealField` and
:class:`~pmesh.pm.ComplexField` objects using the :func:`~pmesh.pm.RealField.r2c`
function for forward FFTs and the :func:`~pmesh.pm.ComplexField.c2r` function
for inverse FFTs. These operations take advantage of the fact that the field objects in
configuration space store real-valued quantities to perform real-to-complex
FFTs. This type of FFT uses the symmetry of real-valued quantities to store
only half of the complex modes along the ``z`` axis.

.. ipython:: python

    # perform the forward FFT
    cfield = rfield.r2c()

    # stores Nmesh/2+1 in z axis b/c of conjugate symmetry
    print(cfield.shape)

    # k=0 mode is the mean value of configuration space field
    print("mean of configuration space field = ", cfield[0,0,0])

    # perform the inverse FFT
    rfield2 = cfield.c2r()

    # print the mean of the underlying array
    print(rfield2.value.mean())

Storing Meta-data
-----------------

For all :class:`MeshSource` objects, the input parameters and additional
meta-data are stored in the :attr:`~MeshSource.attrs` dictionary attribute.

API
---

For more information about specific mesh sources, please see the
:ref:`API section<api-mesh-data>`.
