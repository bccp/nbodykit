.. currentmodule:: nbodykit.algorithms.fftpower

.. _fftpower:

.. ipython:: python
    :suppress:

    import tempfile, os
    startdir = os.path.abspath('.')
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

Simulation Box Power Spectrum/Multipoles (:class:`FFTPower`)
============================================================

The :class:`FFTPower` class computes the 1d power spectrum :math:`P(k)`, 2d
power spectrum :math:`P(k,\mu)`, and/or multipoles :math:`P_\ell(k)` for data
in a simulation box, using a Fast Fourier Transform (FFT). Here, we provide
a brief overview of the algorithm itself as well as the key things to know for
the user to get up and running quickly.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. note::

    See this `cookbook recipe <../cookbook/fftpower.html>`__ for a detailed
    walk-through of the :class:`FFTPower` algorithm.

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
   the field in Fourier space. See :ref:`catalog-to-mesh` for more details.

#. **FFT the mesh to Fourier space**

   Once the density field is painted the mesh, the Fourier transform
   of the field :math:`\delta(\mathbf{x})` is performed in parallel to obtain the complex
   modes of the overdensity field, :math:`\delta(\mathbf{k})`. The field is stored using
   the :class:`~pmesh.pm.ComplexField` object.

#. **Generate the 3D power spectrum on the mesh**

   The 3D power spectrum field is computed on the mesh, using

   .. math::

      P(\mathbf{k}) = \delta(\mathbf{k}) \cdot \delta^\star(\mathbf{k}),

   where :math:`\delta^\star (\mathbf{k})` is the complex conjugate of :math:`\delta(\mathbf{k})`.

#. **Perform the binning in the specified basis**

   Finally, the 3D power defined on the mesh :math:`P(\mathbf{k})` is binned using
   the basis specified by the user. The available options for binning are:

   - 1D binning as a function of wavenumber :math:`k`
   - 2D binning as a function of wavenumber :math:`k` and cosine of the angle to the
     line-of-sight :math:`\mu`
   - Multipole binning as a function of :math:`k` and multipole number :math:`\ell`

The Functionality
-----------------

Users can compute the various quantities using the :class:`FFTPower`:

Auto power spectra and cross power spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both auto and cross spectra are supported. Users can compute cross power spectra
by passing a second mesh object to the :class:`FFTPower` class using
the ``second`` keyword. The first mesh object should always be specified as
the ``first`` argument.

1D Power Spectrum, :math:`P(k)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 1D power spectrum :math:`P(k)` can be computed by specifying the
``mode`` argument as "1d". The wavenumber binning will be linear, and can be
customized by specifying the ``dk`` and ``kmin`` attributes. By default,
the edge of the last wavenumber bin is the
`Nyquist frequency <https://en.wikipedia.org/wiki/Nyquist_frequency>`__, given
by :math:`k_\mathrm{Nyq} = \pi N_\mathrm{mesh} / L_\mathrm{box}`. If ``dk``
is not specified, then the fundamental mode of the box is used:
:math:`2\pi/L_\mathrm{box}`.

2D Power Spectrum, :math:`P(k,\mu)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2D power spectrum :math:`P(k,\mu)` can be computed by specifying the
``mode`` argument as "2d". The number of :math:`\mu` bins is specified via
the ``Nmu`` keyword. The bins range from :math:`\mu=0` to :math:`\mu=1`.

Multipoles of :math:`P(k,\mu)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`FFTPower` class can also compute the multipoles of the 2D power
spectrum, defined as

.. math::

    P_\ell(k) = (2\ell + 1) \int_0^1 d\mu P(k,\mu) \mathcal{L}_\ell(\mu),

where :math:`\mathcal{L}_\ell` is the Legendre polynomial of order
:math:`\ell`. Users can specify which multipoles they wish to compute
by passing a list of the desired :math:`\ell` values as the ``poles``
keyword to the :class:`FFTPower` class.

For example, we can compute both :math:`P(k,\mu)` and :math:`P_\ell(k)`
for a uniform catalog of objects using:

.. ipython:: python

    from nbodykit.lab import UniformCatalog, FFTPower

    cat = UniformCatalog(nbar=100, BoxSize=1.0, seed=42)

    r = FFTPower(cat, mode='2d', Nmesh=32, Nmu=5, poles=[0,2,4])

The Results
-----------

The power spectrum results are stored in two attributes of the
initialized :class:`FFTPower` object:
:attr:`~FFTPower.power` and :attr:`~FFTPower.poles`. These attributes are
:class:`~nbodykit.binned_statistic.BinnedStatistic` objects, which
behave like structured numpy arrays and store
the measured results on a coordinate grid defined by the bins.
See :ref:`analyzing-results` for a full tutorial on using
the :class:`BinnedStatistic` class.

The :attr:`~FFTPower.power` attribute stores the following variables:

- k :
    the mean value for each ``k`` bin
- mu : if ``mode=2d``
    the mean value for each ``mu`` bin
- power :
    complex array storing the real and imaginary components of the power
- modes :
    the number of Fourier modes averaged together in each bin

The :attr:`~FFTPower.poles` attribute stores the following variables:

- k :
    the mean value for each ``k`` bin
- power_L :
    complex array storing the real and imaginary components for
    the :math:`\ell=L` multipole
- modes :
    the number of Fourier modes averaged together in each bin

Note that measured power results for bins where ``modes`` is zero (no data points
to average over) are set to ``NaN``.

In our example, the ``power`` and ``poles`` attributes are:

.. ipython:: python

    # the 2D power spectrum results
    print(r.power)
    print("variables = ", r.power.variables)
    for name in r.power.variables:
        var = r.power[name]
        print("'%s' has shape %s and dtype %s" %(name, var.shape, var.dtype))

    # the multipole results
    print(r.poles)
    print("variables = ", r.poles.variables)
    for name in r.poles.variables:
        var = r.poles[name]
        print("'%s' has shape %s and dtype %s" %(name, var.shape, var.dtype))

These attributes also store meta-data computed during the power calculation
in the ``attrs`` dictionary.  Most importantly, the ``shotnoise`` key
gives the Poisson shot noise, :math:`P_\mathrm{shot} = V / N`, where *V*
is the volume of the simulation box and *N* is the number of objects. The keys
``N1`` and ``N2`` store the number of objects

In our example, the meta-data is:

.. ipython:: python

    for k in r.power.attrs:
      print("%s = %s" %(k, str(r.power.attrs[k])))

Saving and Loading
------------------

Results can easily be saved and loaded from disk in a reproducible manner
using the :func:`FFTPower.save` and :class:`FFTPower.load` functions.
The :class:`~FFTPower.save` function stores the state of the algorithm,
including the meta-data in the :attr:`FFTPower.attrs` dictionary, in a
JSON plaintext format.

.. ipython:: python

    # save to file
    r.save("fftpower-example.json")

    # load from file
    r2 = FFTPower.load("fftpower-example.json")

    print(r2.power)
    print(r2.poles)
    print(r2.attrs)

Common Pitfalls
---------------

The default configuration of nbodykit should lead to reasonable results
when using the :class:`FFTPower` algorithm. When performing custom, more complex
analyses, some of the more common pitfalls are:

- When the results of :class:`FFTPower` do not seem to make sense, the most common
  culprit is usually the configuration of the mesh, and whether or not the mesh
  is "compensated". In the language of nbodykit, "compensated" refers to whether
  the effects of the interpolation window used to paint the density field have
  been de-convolved in Fourier space. See the section :class:`catalog-to-mesh`
  for detailed notes on this procedure.

- Be wary of normalization issues when painting weighted density fields. See
  :ref:`mesh-normalization` for the default normalization scheme and :ref:`mesh-apply`
  for notes on applying arbitrary functions to the mesh while painting. The
  section :ref:`weighted-painting` describes the procedure to use when
  painting a weighted density field.

.. ipython:: python
    :suppress:

    import shutil
    os.chdir(startdir)
    shutil.rmtree(tmpdir)
