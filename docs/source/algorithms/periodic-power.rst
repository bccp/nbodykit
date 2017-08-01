.. currentmodule:: nbodykit.algorithms.fftpower

.. _fftpower:

Simulation Box Power Spectrum/Multipoles (:class:`~nbodykit.algorithms.fftpower.FFTPower`)
==========================================================================================

The :class:`FFTPower` class computes the 1d power spectrum :math:`P(k)`, 2d
power spectrum :math:`P(k,\mu)`, and/or multipoles :math:`P_\ell(k)` for data
in a simulation box, using a Fast Fourier Transform (FFT). Here, we provide
a brief overview of the algorithm itself as well as the key things to know for
user to get up and running quickly.

The Algorithm
-------------

The steps involved in computing the power spectrum via :class:`FFTPower`
are as follows:

#. **Generate data on a mesh**


   Data must be painted on to a discrete mesh to compute the power spectrum.
   There are several ways to generate data on a mesh (see :ref:`creating-mesh`),
   but the most common is painting a discrete catalog of objects
   on to a mesh (see :ref:`catalog-to-mesh`). The :class:`FFTPower` class accepts
   input data in either the form of a :class:`~nbodykit.base.mesh.MeshSource`
   or a :class:`~nbodykit.base.catalog.CatalogSource`. In the latter case,
   the catalog is automatically converted to a mesh using the default parameters
   of the :func:`~nbodykit.base.catalog.CatalogSource.to_mesh` function.

   When converting from a catalog to a mesh, users can customize the painting
   procedure via the options of the
   :func:`~nbodykit.base.catalog.CatalogSource.to_mesh` function.
   These options have important effects on the resulting power spectrum of
   the field in Fourier space. See :ref:`setting-mesh-params` for more details.

#. **FFT the mesh to Fourier space**

   Once the density field is painted the mesh, the Fourier transform
   of the field :math:`\delta(\mathbf{x})` is performed in parallel to obtain the complex
   modes of the overdensity field, :math:`\delta(\mathbf{k})`. The field is stored using
   the :class:`~pmesh.pm.ComplexField` object.

#. **Generate the 3D power spectrum on the mesh**

   The 3D power spectrum field is computed on the mesh, using

   .. math::

      P(\mathbf{k}) = \delta(\mathbf{k}) * \delta^\star(\mathbf{k}),

   where :math:`\delta^\star (\mathbf{k})` is the complex conjugate of :math:`\delta(\mathbf{k})`.

#. **Perform the binning in the specified basis**

The Functionality
-----------------

The Results
-----------

Saving and Loading
------------------

Common Pitfalls
---------------

.. _projected-fftpower:

Power Spectrum of a Projected Field (:class:`~nbodykit.algorithms.fftpower.ProjectedFFTPower`)
==============================================================================================
