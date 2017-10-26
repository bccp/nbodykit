.. currentmodule:: nbodykit.algorithms.convpower

.. _convpower:

Power Spectrum Multipoles of Survey Data (:class:`ConvolvedFFTPower`)
=====================================================================

The :class:`ConvolvedFFTPower` class computes the power spectrum multipoles
:math:`P_\ell(k)` for data from an observational survey that includes non-trivial
selection effects. The input data is expected to be in the form of
angular coordinates (right ascension and declination) and redshift.
The measured power spectrum multipoles represent the true multipoles convolved
with the window function. Here, the window function refers to the Fourier
transform of the survey volume. In this section, we provide an overview of the
FFT-based algorithm used to compute the multipoles
and detail important things to know for the user to get up and running quickly.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. note::

    To jump right into the :class:`ConvolvedFFTPower` algorithm, see this
    `cookbook recipe <cookbook/convpower.ipynb>`_ for a detailed
    walk-through of the algorithm.

.. _fkp-background:

Some Background
---------------

We begin by defining the estimator for the multipole power spectrum,
often referred to as the "Yamamoto estimator" in the literature
(`Yamamoto et al 2006`_):

.. math::
    :label: Pell

    P_\ell = \frac{2\ell+1}{A} \int \frac{\d\Omega_k}{4\pi}
              \left [
              \int \d\vrone \ F(\vrone) e^{i \vk \cdot \vrone}
              \int \d\vrtwo \ F(\vrtwo)
                                        e^{-i \vk \cdot \vrtwo}
                                        \L_\ell(\vkhat \cdot \vrhat_2)
                                        - P_\ell^{\rm noise}(\vk)
                                        \right ].

where :math:`\Omega_k` represents the solid angle in Fourier space,
and :math:`\L_\ell` is the Legendre polynomial of
order :math:`\ell`. The weighted density field :math:`F(\vr)` is defined as

.. math::
    :label: F

    F(\vr) = \wfkp(\vr) \left[ n_g'(\vr) - \alpha' \ n'_s(\vr)\right],


where :math:`n_g'` and :math:`n_s'` are the number densities for the
galaxy catalog and synthetic catalog of randoms object, respectively, and
:math:`\alpha'` is the ratio of the number of real galaxies to random galaxies.
Often, the catalog of random objects has a much higher number density than the
galaxy catalog, and the factor of :math:`\alpha'` re-normalizes to the proper
number density. The normalization :math:`A` is given by

.. math::
    :label: A-integral

    A = \int \d\vr [n'_g(\vr)  \wfkp(\vr)]^2.

The shot noise :math:`P_\ell^{\rm noise}` in equation :eq:`Pell` is

.. math::
    :label: Pshot

    P_\ell^{\rm noise}(\vk) = (1 + \alpha') \int \d\vr \ n'_g(\vr) \wfkp^2(\vr) \L_\ell (\vkhat \cdot \vrhat).

The FKP weights, first derived in
`Feldman, Kaiser, and Peacock 1994`__, minimize the variance of the estimator
at a desired power spectrum value. Denoted as
:math:`w_\mathrm{FKP}`, these weights are given by

.. math::
    :label: wfkp

    w_\mathrm{FKP}(\vr) = \frac{1}{1 + n_g'(\vr) P_0},

where :math:`P_0` is the power spectrum amplitude in units of
:math:`h^{-3} \mathrm{Mpc}^3` where the estimator is optimized. For typically
galaxy survey analyses, a value of order :math:`P_0 = 10^{4} \ h^{-3} \mathrm{Mpc}^3`
is usually assumed.

In our notation, quantities marked with a prime (:math:`'`) include
completeness weights. These weights, denoted as :math:`w_c`,
help account for systematic variations in the number density fields. Typically
the weights for the random catalog are unity, but for full generality we
allow for the possibility of non-unity weights for :math:`n_s` as well.
For example, :math:`\alpha' = N'_\mathrm{gal} / N'_\mathrm{ran}`, where
:math:`N'_\mathrm{gal} = \sum_i^{N_\mathrm{gal}} w_c^i`.

.. _fkp-algorithm:

The Algorithm
-------------

We use the FFT-based algorithm of `Hand et al 2017`_ to compute the power spectrum
multipoles. This work improves upon the previous FFT-based estimators of
:math:`P_\ell(k)` presented in `Bianchi et al 2015`_ and `Scoccimarro 2015`_
by only requiring the use of :math:`2\ell+1` FFTs to compute a multipole of
order :math:`\ell`. The algorithms uses the spherical harmonic addition
theorem to express equation :eq:`Pell` as

.. math::
    :label: Pell-ours

    P_\ell(k) = \frac{2\ell+1}{A} \int \frac{\d\Omega_k}{4\pi} F_0(\vk) F_\ell(-\vk),

where

.. math::
    :label: Fell

    F_\ell(\vk) &= \int \d\vr \ F(\vr) e^{i \vk \cdot \vr}
                \L_\ell(\vkhat \cdot \vrhat),

                &= \frac{4\pi}{2\ell+1} \sum_{m=-\ell}^{\ell}
                    \Ylm(\vkhat) \int \d\vr \ F(\vr) \Ylm^*(\vrhat) e^{i \vk \cdot \vr}.

The sum over :math:`m` in equation :eq:`Fell` contains :math:`2 \ell + 1`
terms, each of which can be computed using a FFT. Thus, the multipole moments
can be expressed as a sum of Fourier transforms of the weighted density field,
with weights given by the appropriate spherical harmonic.
We evaluate equation :eq:`Fell` using the real-to-complex FFT functionality
of the :mod:`pmesh` package and
use the `real form of the spherical harmonics`_ :math:`\Ylm`.

We use the symbolic manipulation functionality
available in the `SymPy`_ Python package to compute the
spherical harmonic expressions in terms
of Cartesian vectors. This allows the user to specify the desired multipoles
at runtime, enabling the code to be used to compute multipoles of arbitrary
:math:`\ell`.

.. _fkp-getting-started:

Getting Started
---------------

Here, we outline the necessary steps for users to get started using the
:class:`ConvolvedFFTPower` algorithm to compute the power spectrum multipoles
from an input data catalog.

The ``FKPCatalog`` Class
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ConvolvedFFTPower` algorithm requires a galaxy catalog and a
synthetic catalog of random objects without any clustering signal,
which we refer to as the "data" and "randoms" catalogs, respectively.
The :class:`CatalogSource` object responsible for handling these two types
of catalogs is the :class:`~nbodykit.source.catalog.fkp.FKPCatalog` class.

The :class:`~nbodykit.source.catalog.fkp.FKPCatalog` determines the size of
the Cartesian box that the  "data" and "randoms" are placed in, which is then
also used during the FFT operation. By default, the box size is determined
automatically from the maximum extent of the "randoms" positions.
In this automatic case, the size of the box can be artificially extended and
padded with zeros via the ``BoxPad`` keyword.
Users can also specify a desired box size by passing in the ``BoxSize`` keyword.

The :class:`~nbodykit.source.catalog.fkp.FKPCatalog`
object can be converted to a mesh object,
:class:`~nbodykit.source.catalogmesh.fkp.FKPCatalogMesh`, via the
:func:`~nbodykit.source.catalog.fkp.FKPCatalog.to_mesh` function. This
mesh object knows how to paint the FKP weighted density field,
given by equation :eq:`F`, to the mesh using the "data" and "randoms" catalogs.
With the FKP density field painted to the mesh, the :class:`ConvolvedFFTPower`
algorithm uses equations :eq:`Pell-ours` and :eq:`Fell` to compute the
multipoles specified by the user.

From Sky to Cartesian Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Position`` column, holding the Cartesian coordinates,
is required for both the "data" and "randoms" catalogs.
We provide the function :func:`nbodykit.transform.SkyToCartesian` for converting
sky coordinates, in the form of right ascension, declination, and redshift,
to Cartesian coordinates. The conversion from redshift to comoving distance
requires a cosmology instance, which can be specified via the
:class:`~nbodykit.cosmology.core.Cosmology` class.

For more details, see :ref:`sky-to-cartesian`.

.. _fkp-nbar:

Specifying :math:`n'_g(z)`
^^^^^^^^^^^^^^^^^^^^^^^^^^

The number density of the "data" catalog as a function of redshift, in units
of :math:`h^{3} \mathrm{Mpc}^{-3}`, is required to properly normalize the
power spectrum using equation :eq:`A-integral` and to compute the shot noise
via equation :eq:`Pshot`.  The "data" and "randoms" catalog should contain a
column that gives this quantity, evaluated at the redshift of each object in
the catalogs.

When converting from a :class:`~nbodykit.source.catalog.fkp.FKPCatalog`
to a :class:`~nbodykit.source.catalogmesh.fkp.FKPCatalogMesh`, the name
of the :math:`n'_g(z)` column should be passed as the ``nbar`` keyword to the
:func:`~nbodykit.source.catalog.fkp.FKPCatalog.to_mesh` function. The
:math:`n'_g(z)` column should have the same name in the "data" and "randoms"
catalogs.

By default, the name of the ``nbar`` column is set to ``NZ``. If this
column is missing from the "data" or "randoms" catalog, an exception
will be raised.

Note that the :class:`~nbodykit.algorithms.zhist.RedshiftHistogram` algorithm
can compute a weighted :math:`n(z)` for an input catalog and may be useful
if the user needs to compute :math:`n_g(z)`.

.. important::

    By construction, the objects in the "randoms" catalog should follow the
    same redshift distribution as the "data" catalog. Often, the "randoms"
    will have an overall number density that is 10, 50, or 100 times the number
    density of the "data" catalog. Even if the "randoms" catalog has a higher
    number density, the ``nbar`` column in both the "data" and "randoms"
    catalogs should hold number density values on the same scale, corresponding
    to the value of :math:`n_g(z)` at the redshift of the objects in the
    catalogs.

Using Completeness Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ConvolvedFFTPower` algorithm supports the use of completeness
weights to account for systematic variations in the number density of the
"data" and "randoms" objects. In our notation in the earlier
:ref:`background section <fkp-background>`, quantities marked with a
prime (:math:`'`) include completeness weights. The weighted and
unweighted number densities of the "data" and "randoms" fields are
then related by:

.. math::

    n'_g(\vr) &= w_{c,g}(\vr) n_g(\vr),

    n'_s(\vr) &= w_{c,s}(\vr) n_s(\vr),

Typically the weights for the random catalog are unity, but for full generality,
we allow for the possibility of non-unity weights for :math:`n_s` as well.
For example, :math:`\alpha' = N'_\mathrm{gal} / N'_\mathrm{ran}`, where
:math:`N'_\mathrm{gal} = \sum_i^{N_\mathrm{gal}} w_{c,g}^i` and
:math:`N'_\mathrm{ran} = \sum_i^{N_\mathrm{ran}} w_{c,s}^i`.

When converting from a :class:`~nbodykit.source.catalog.fkp.FKPCatalog`
to a :class:`~nbodykit.source.catalogmesh.fkp.FKPCatalogMesh`, the name
of the completeness weight column should be passed as the ``comp_weight``
keyword to the :func:`~nbodykit.source.catalog.fkp.FKPCatalog.to_mesh` function.

By default, the name of the ``comp_weight`` column is set to the
default column ``Weight``, which has a value of unity for all objects.
If specifying a different name, the column should have the same name in both
the "data" and "randoms" catalogs.

Using FKP Weights
^^^^^^^^^^^^^^^^^

Users can also specify the name of a column in the "data" and "randoms"
catalogs that represents a FKP weight for each object, as given by
equation :eq:`wfkp`. The FKP weights do not weight the individual number
density fields as the completeness weights do, but rather they weight
the combined field, :math:`n'_g(\vr) - \alpha' n'_s(\vr)` (see equation
:eq:`F`).

When converting from a :class:`~nbodykit.source.catalog.fkp.FKPCatalog`
to a :class:`~nbodykit.source.catalogmesh.fkp.FKPCatalogMesh`, the name
of the FKP weight column should be passed as the ``fkp_weight``
keyword to the :func:`~nbodykit.source.catalog.fkp.FKPCatalog.to_mesh` function.

By default, the name of the ``fkp_weight`` column is set to ``FKPWeight``,
which has a value of unity for all objects.  If specifying a different name,
the column should have the same name in both the "data" and "randoms"
catalogs.

The :class:`ConvolvedFFTPower` algorithm can also automatically generate
and use FKP weights from the input ``nbar`` column if the user specifies
the ``use_fkp_weights`` keyword of the algorithm to be ``True``. In this case,
the user must also specify the ``P0_FKP`` keyword, which gives the desired
:math:`P_0` value to use in equation :eq:`wfkp`.


The Results
-----------

The Multipoles
^^^^^^^^^^^^^^

The multipole results are stored as the :attr:`~ConvolvedFFTPower.poles`
attribute of the initialized :class:`ConvolvedFFTPower` object.
This attribute is a :class:`~nbodykit.binned_statistic.BinnedStatistic` object,
which behaves like a structured numpy array and stores
the measured results on a coordinate grid defined by the wavenumber bins
specified by the user. See :ref:`analyzing-results` for a full tutorial on using
the :class:`BinnedStatistic` class.

The :attr:`~ConvolvedFFTPower.poles` attribute stores the following variables:

- k :
    the mean value for each ``k`` bin
- power_L :
    complex array storing the real and imaginary components for
    the :math:`\ell=L` multipole
- modes :
    the number of Fourier modes averaged together in each bin

Note that measured power results for bins where ``modes`` is zero (no data points
to average over) are set to ``NaN``.

.. note::

    The shot noise is not subtracted from any measured results. Users can
    access the shot noise value for monopole, computed according to
    equation :eq:`Pshot` in the meta-data :attr:`attrs` dictionary. Shot
    noise for multipoles with :math:`\ell>0` is assumed to be zero.

.. _fkp-meta-data:

The Meta-data
^^^^^^^^^^^^^

Several important meta-data calculations are also performed during the
algorithm's execution. This meta-data is stored in both the
:attr:`ConvolvedFFTPower.attrs` attribute and the :attr:`ConvolvedFFTPower.poles.attrs`
atrribute.

#. data.N, randoms.N :
    the unweighted number of data and randoms objects
#. data.W, randoms.W :
    the weighted number of data and randoms objects, using the
    column specified as the completeness weights. This is given by:

    .. math::

        W_\mathrm{data} &= \sum_\mathrm{data} w_c

        W_\mathrm{ran} &= \sum_\mathrm{randoms} w_c

#. alpha :
    the ratio of ``data.W`` to ``randoms.W``
#. data.norm, randoms.norm :
    the normalization :math:`A` of the power spectrum, computed from either
    the "data" or "randoms" catalog (they should be similar).
    They are given by:

    .. math::
          :label: A-sum

          A_\mathrm{data} &= \sum_\mathrm{data} n'_g w_c \wfkp^2

          A_\mathrm{ran} &= \alpha \sum_\mathrm{randoms} n'_g w_c \wfkp^2

#. data.shotnoise, randoms.shotnoise :
    the contributions to the monopole shot noise from the "data" and "random"
    catalogs. These values are given by:

    .. math::

        P^\mathrm{shot}_\mathrm{data} &= A_\mathrm{ran}^{-1} \sum_\mathrm{data} (w_c \wfkp)^2

        P^\mathrm{shot}_\mathrm{ran} &= \alpha^2 A_\mathrm{ran}^{-1} \sum_\mathrm{randoms} (w_c \wfkp)^2

#. shotnoise :
    the total shot noise for the monopole power spectrum, which should be
    subtracted from the monopole measurement in :attr:`ConvolvedFFTPower.poles`.
    This is computed as:

    .. math::

        P^\mathrm{shot} = P^\mathrm{shot}_\mathrm{data} + P^\mathrm{shot}_\mathrm{ran}

#. BoxSize :
    the size of the Cartesian box used to grid the "data" and
    "randoms" objects on the Cartesian mesh.

From :math:`P_\ell(k)` to :math:`P(k,\mu)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`ConvolvedFFTPower.to_pkmu` allows users to rotate the measured
multipoles, stored as the :attr:`ConvolvedFFTPower.poles` attribute, into
:math:`P(k,\mu)` wedges (bins in :math:`\mu`). The function returns a
:class:`~nbodykit.binned_statistic.BinnedStatistic` holding the
binned :math:`P(k,\mu)` data.


Saving and Loading
------------------

Results can easily be saved and loaded from disk in a reproducible manner
using the :func:`ConvolvedFFTPower.save` and :func:`ConvolvedFFTPower.load`
functions. The :class:`~ConvolvedFFTPower.save` function stores the state of
the algorithm, including the meta-data in the :attr:`ConvolvedFFTPower.attrs`
dictionary, in a JSON plaintext format.

Common Pitfalls
---------------

Some of the more common issues users run into are:

- The ``Position`` column is missing from the input "data" and "randoms" catalogs.
  The input position columns for this algorithm are assumed to be in terms
  of the right ascension, declination, and redshifts, and users must add
  the ``Position`` column holding the Cartesian coordinates explicitly to both
  the "data" and "randoms" catalogs.

- Normalization issues may occur if the number density columns in the "data"
  and "randoms" catalogs are on different scales. Similar issues may arise
  if the FKP weight column uses number density values on different scales. In
  all cases, the number density to be used should be that of the data, denoted
  as :math:`n'_g(z)`. The algorithm will compute the power spectrum normalization
  from both the "data" and "randoms" catalogs (as given by equation :eq:`A-sum`).
  If the values do not agree, there is likely an issue with varying
  number density scales, and the algorithm will raise an exception.


.. _Hand et al 2017: https://arxiv.org/abs/1704.02357
.. _Bianchi et al 2015: https://arxiv.org/abs/1505.05341
.. _Scoccimarro 2015: https://arxiv.org/abs/1506.02729
.. _FKP: https://arxiv.org/abs/astro-ph/9304022
__ FKP_
.. _Yamamoto et al 2006: https://arxiv.org/abs/astro-ph/0505115
.. _real form of the spherical harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
.. _SymPy: http://www.sympy.org
