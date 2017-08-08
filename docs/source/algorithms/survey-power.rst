.. currentmodule:: nbodykit.algorithms.convpower

.. _convpower:

.. ipython:: python
    :suppress:

    import tempfile, os
    startdir = os.path.abspath('.')
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

Power Spectrum Multipoles of Survey Data (:class:`ConvolvedFFTPower`)
=====================================================================


The :class:`ConvolvedFFTPower` class computes the power spectrum multipoles
:math:`P_\ell(k)` for data from a survey that includes non-trivial
selection effects. The input data is expected to be in the form of
angular coordinates (right ascension and declination) and redshift.
The measured power spectrum multipoles represent the true multipoles convolved
with the window function. Here, the window function refers to the the Fourier
transform of the survey volume. In this section, we provide an overview of the
FFT-based algorithm used to compute the multipoles
and detail important things to know for the user to get up and running quickly.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. note::

    To jump right into the :class:`ConvoledFFTPower` algorithm, see this
    `cookbook recipe <../cookbook/convpower.html>`_ for a detailed
    walk-through of the :class:`ConvolvedFFTPower` algorithm.

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
The normalization :math:`A` is given by

.. math::
    :label: A-integral

    A = \int \d\vr [n'_g(\vr)  \wfkp(\vr)]^2.

The shot noise :math:`P_\ell^{\rm noise}` in equation :eq:`Pell` is

.. math::
    :label:  Pshot

    P_\ell^{\rm noise}(\vk) = (1 + \alpha') \int \d\vr \ \bar{n}(\vr) \wfkp^2(\vr) \L_\ell (\vkhat \cdot \vrhat).

The FKP weights, first derived in
`Feldman, Kaiser, and Peacock 1994`__, minimize the variance of the estimator
at a desired power spectrum value. Denoted as
as :math:`w_\mathrm{FKP}`, these weights are given by

.. math::
    :label: wfkp

    w_\mathrm{FKP}(\vr) = \frac{1}{1 + n_g'(\vr) P_0},

where :math:`P_0` is the power spectrum amplitude in units of
:math:`h^{-3} \mathrm{Mpc}^3` where the estimator is optimized. For typically
galaxy survey analyses, a value of order :math:`P_0 = 10^{4} \ h^{-3} \mathrm{Mpc}^3`
is assumed.

In our notation, quantities marked with a prime (:math:`'`) include
completeness weights. These weights, denoted as :math:`w_c`,
help account for systematic variations in the number density fields. Typically
the weights for the random catalog are unity, but for full generality we
allow for the possibility of non-unity weights for :math:`n_s` as well.
For example, :math:`\alpha' = N'_\mathrm{gal} / N'_\mathrm{ran}`, where
:math:`N'_\mathrm{gal} = \sum_i^{N_\mathrm{gal}} w_c,i`.

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

    P_\ell(k) = \frac{2\ell+1}{I} \int \frac{\d\Omega_k}{4\pi} F_0(\vk) F_\ell(-\vk),

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
available in the \texttt{sympy} Python package \cite{sympy} to compute the
spherical harmonic expressions in equation~\ref{eq:real-Ylm} in terms
of Cartesian vectors. This allows the user to specify the desired multipoles
at runtime, enabling the code to be used to compute multipoles of arbitrary $\ell$.
Testing and development of the code was performed on the
Cray XC-40 system Cori at the National Energy Research Supercomputing Center (NERSC),
and the code exhibits strong scaling, with a roughly linear reduction in wall-clock
time as the number of available processors increases. When computing all even
multipoles up to $\lmax = 16$ (requiring in total 153 FFTs), our
implementation takes roughly 90 seconds using 64 processors on Cori.

For the results presented in this work, we place the galaxies and random objects
on a Cartesian grid using the Triangular Shaped Cloud (TSC) prescription
to compute the density field $F(\vr)$ of equation~\ref{eq:FKP-density}. We use the
interlaced grid technique of \cite{SefusattiEtAl16} to limit the effects of
aliasing, and we correct for any artifacts of the TSC gridding using the
correction factor of \cite{Jing05}. The interlacing scheme
allows computation of the FFTs on a $512^3$ grid with accuracy comparable
to the results when using a $1024^3$ grid, but with a wall-clock time that is
$\sim8\times$ smaller. When using interlacing, the catalog of galaxies is
interpolated on to two meshes separated by half of the size of a grid cell.
We sum these two density fields in Fourier space and inverse Fourier Transform
back to configuration space. We then apply the spherical harmonic weightings of
equation~\ref{eq:real-Ylm} to this combined density field
and proceed with computing the terms in equation~\ref{eq:Fell}.
The speed-up provided by interlacing is particularly
powerful when computing large $\ell$ multipoles. When combined with TSC
interpolation, we are able to measure power spectra up to the
Nyquist frequency at $k \simeq 0.4 \ h \mathrm{Mpc}^{-1}$ with
fractional errors at the level of $10^{-3}$ \cite{SefusattiEtAl16}.

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
`Nyquist frequency <https://en.wikipedia.org/wiki/Nyquist_frequency>`_, given
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

.. _Hand et al 2017: https://arxiv.org/abs/1704.02357
.. _Bianchi et al 2015: https://arxiv.org/abs/1505.05341
.. _Scoccimarro 2015: https://arxiv.org/abs/1506.02729
.. _FKP: https://arxiv.org/abs/astro-ph/9304022
__ FKP_
.. _Yamamoto et al 2006: https://arxiv.org/abs/astro-ph/0505115
.. _real form of the spherical harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
