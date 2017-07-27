.. _common-mesh-operations:

.. currentmodule:: nbodykit.base.mesh

.. ipython:: python
    :suppress:

    import tempfile, os
    startdir = os.path.abspath('.')
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

Common Mesh Operations
======================

In this section, we detail some common operations for manipulating data
defined on a mesh via a :class:`MeshSource` object.


.. _mesh-preview:

Previewing the Mesh
-------------------

The :func:`MeshSource.preview` function allows users to preview a
low-resolution mesh by resampling the mesh and gathering the mesh to all
ranks. It can also optionally project the mesh across multiple axes, which
enables visualizations of the projected density field for quick data
inspection by the user.

For example, below we initialize a
:class:`~nbodykit.source.mesh.linear.LinearMesh` object on a :math:`128^3`
mesh and preview the mesh on a :math:`64^3` mesh after projecting the field
along two axes:

.. ipython:: python

    from nbodykit.lab import LinearMesh, cosmology
    from pylab import *

    cosmo = cosmology.Planck15
    Plin = cosmology.EHPower(cosmo, redshift=0)

    mesh = LinearMesh(Plin, Nmesh=128, BoxSize=1380, seed=42)

    density = mesh.preview(Nmesh=64, axes=(0,1))
    @savefig density-preview.png
    imshow(density)

.. _saving-loading-mesh:

.. note::

    The previewed mesh result is broadcast to all ranks, so each rank
    allocates :math:`\mathrm{Nmesh}^3` in memory.

Saving and Loading a Mesh
-------------------------

The :func:`MeshSource.save` function paints the mesh via a call to
:func:`MeshSource.paint` and saves the mesh using a :mod:`bigfile` format.
The output mesh saved to file can be in either configuration space or Fourier
space, by specifying ``mode`` as either ``real`` or ``complex``.

Below, we save our :class:`~nbodykit.source.mesh.linear.LinearMesh` to a
:mod:`bigfile` file:

.. ipython:: python

    # save the RealField
    mesh.save('linear-mesh-real.bigfile', mode='real', dataset='Field')

    # save the ComplexField
    mesh.save('linear-mesh-complex.bigfile', mode='real', dataset='Field')

The saved mesh can be loaded from disk using the
:class:`~nbodykit.source.mesh.bigfile.BigFileMesh` class:

.. ipython:: python

    from nbodykit.lab import BigFileMesh
    import numpy

    # load the mesh in the form of a RealField
    real_mesh = BigFileMesh('linear-mesh-real.bigfile', 'Field')

    # return the RealField via paint
    rfield = real_mesh.paint(mode='real')

    # load the mesh in the form of a ComplexField
    complex_mesh = BigFileMesh('linear-mesh-complex.bigfile', 'Field')

    # FFT to get the ComplexField as a RealField
    rfield2 = complex_mesh.paint(mode='real')

    # the two RealFields must be the same!
    numpy.allclose(rfield.value, rfield2.value)

Here, we load our meshes in configuration space and Fourier space and then
paint both with ``mode=real`` and verify that the results are the same.

.. _mesh-apply:

Applying Functions to the Mesh
------------------------------

nbodykit supports performing transformations to the mesh data by applying
arbitrary functions in either configuration space or Fourier space. Users
can use the :func:`MeshSource.apply` function to apply these transformations.
The function applied to the mesh should take two arguments, ``x`` and ``v``:

#. The ``x`` argument provides a list of length three holding the coordinate
   arrays that define the mesh. These arrays broadcast to the full shape of the
   mesh, i.e., they have shapes :math:`(N_x,1,1)`, :math:`(1,N_y,1)`, and
   :math:`(1,1,N_z)` if the mesh has shape :math:`(N_x, N_y, N_z)`.
#. The ``v`` argument is the array holding the value of the mesh field at
   the coordinate arrays in ``x``

The units of the ``x`` coordinate arrays depend upon the values of the
``kind`` and ``mode`` keywords passed to the :func:`~MeshSource.apply` function.
The various cases are:

=========== ============== ==============================
mode        kind           range of ``x`` argument
``real``    ``relative``   :math:`[-L/2, L/2)`
``real``    ``index``      :math:`[0, N)`
``complex`` ``wavenumber`` :math:`[- \pi N/L, \pi N / L)`
``complex`` ``circular``   :math:`[-\pi, \pi)`
``complex`` ``index``      :math:`[0, N)`
=========== ============== ==============================

Here, :math:`L` is the size of the box and `N` is the number of cells per mesh side.

In the example below, we apply a filter function in Fourier space that divides
the mesh by the squared norm of the wavenumber ``k`` on the mesh, and then
print out the first few mesh cells of the filtered mesh to verify the
function was applied properly.

.. ipython:: python

    def filter(k, v):
        kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
        kk[kk == 0] = 1
        return v / kk # divide the mesh by k^2

    # apply the filter and get a new mesh
    filtered_mesh = mesh.apply(filter, mode='complex', kind='wavenumber')

    # get the filtered RealField object
    filtered_rfield = filtered_mesh.paint(mode='real')

    print("head of filtered Realfield = ",  filtered_rfield[:10,0,0])
    print("head of original RealField = ",  rfield[:10,0,0])


.. _mesh-resample:

Resampling a Mesh
-----------------

Users can resample a mesh to by painting the mesh to either a
:class:`~pmesh.pm.RealField` or :class:`~pmesh.pm.ComplexField` and
initializing a new :class:`MemoryMesh` object with a different value for ``Nmesh``.
For example, below we resample the our original
:class:`~nbodykit.source.mesh.linear.LinearMesh` object, changing the mesh
resolution from ``Nmesh=128`` to ``Nmesh=32``.

.. ipython:: python

    from nbodykit.lab import MemoryMesh

    print("original Nmesh = ", mesh.attrs['Nmesh'])

    # the original complex field
    cfield = mesh.paint(mode='complex')

    # initialize a re-sampled mesh
    resampled_mesh = MemoryMesh(cfield, Nmesh=32)

    print("new Nmesh = ", resampled_mesh.attrs['Nmesh'])

.. ipython:: python
    :suppress:

    import shutil
    os.chdir(startdir)
    shutil.rmtree(tmpdir)
