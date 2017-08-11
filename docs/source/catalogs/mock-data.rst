.. _mock-catalogs:

.. currentmodule:: nbodykit.source.catalog

Generating Catalogs of Mock Data
================================

nbodykit includes several methods for generating mock catalogs, with varying
levels of sophistication. These :class:`~nbodykit.base.catalog.CatalogSource`
objects allow users to create catalogs of objects at run time and include:

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. _random-mock-data:

Randomly Distributed Objects
----------------------------

nbodykit includes two subclasses of
:class:`~nbodykit.base.catalogs.CatalogSource` that
generate particles randomly in a box: :class:`~uniform.RandomCatalog`
and :class:`~uniform.UniformCatalog`. While these catalogs do not produce
realistic cosmological distributions of objects, they are especially useful
for generating catalogs quickly and for testing purposes.

.. _random-catalog:

:class:`~uniform.RandomCatalog`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~uniform.RandomCatalog` class includes a random number generator
with the functionality of :class:`numpy.random.RandomState` that generates
random numbers in parallel and in a manner that is independent of the number
of MPI ranks being used. This property is especially useful for running
reproducible tests where the number of CPUs might vary. The random number
generator is stored as the :attr:`~uniform.RandomCatalog.rng` attribute.
Users can use this random number generator to add columns to the catalog,
using the :ref:`syntax to add columns <adding-columns>`.

For example,

.. ipython:: python

    from nbodykit.lab import RandomCatalog
    import numpy

    # initialize a catalog with only the default columns
    cat = RandomCatalog(csize=100) # collective size of 100
    print("columns = ", cat.columns) # only the default columns present

    # add mass uniformly distributed in log10
    cat['Mass'] = 10**(cat.rng.uniform(12, 15, size=cat.size))

    # add normally distributed velocity
    cat['Velocity'] = cat.rng.normal(loc=0, scale=500, size=cat.size)

    print(cat.columns)

**Caveats**

- For a list of the full functionality of the :attr:`~uniform.RandomCatalog.rng`
  attribute, please see the API documentation for :class:`~uniform.MPIRandomState`.
- When adding columns, the new column must have the same length as the local
  size of the catalog, as specified by the :attr:`size` attribute. Most functions
  of the :attr:`~uniform.RandomCatalog.rng` attribute accept the ``size``
  keyword to generate an array of the correct size.


.. _uniform-catalog:

:class:`~uniform.UniformCatalog`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~uniform.UniformCatalog` is a subclass of
:class:`~uniform.RandomCatalog` that includes ``Position`` and ``Velocity``
columns that are uniformly distributed. The positions of the particles are
uniformly distributed between zero and the size of the box (as specified by
the user), with an input number density. The velocities are also
uniformly distributed but on a scale that is 1% of the size of the box.

For example,

.. ipython:: python

  from nbodykit.lab import UniformCatalog

  cat = UniformCatalog(nbar=100, BoxSize=1.0, seed=42)
  print("columns = ", cat.columns)

  # min must be greater than 0
  print("position minimum = ", cat.compute(cat['Position'].min()))

  # max must be less than 1.0
  print("position maximum = ", cat.compute(cat['Position'].max()))

  # min must be greater than 0
  print("velocity minimum = ", cat.compute(cat['Velocity'].min()))

  # max must be less than 0.01
  print("velocity maximum = ", cat.compute(cat['Velocity'].max()))

Note that because :class:`~uniform.UniformCatalog` is a subclass of
:class:`~uniform.RandomCatalog` users can also use the
:attr:`~uniform.UniformCatalog.rng` attribute to add new columns to
a :class:`~uniform.UniformCatalog` object.

.. _lognormal-mock-data:

Log-normal Mocks
----------------

The :class:`~lognormal.LogNormalCatalog` offers a more realistic
approximation of cosmological large-scale structure. The class
generates a set of objects by Poisson sampling a log-normal density field,
using the Zel'dovich approximation to model non-linear evolution.
Given a linear power spectrum function, redshift, and linear bias
supplied by the user, this class performs the following steps:

**Generate Gaussian initial conditions**

First, a Gaussian overdensity field :math:`\delta_L(\vk)` is generated in
Fourier space with a power spectrum given by a function specified by
the user. We also generate linear velocity fields from the overdensity field
in Fourier space, using

.. math::

    \vv(\vk) = i f a H \delta_L(\vk) \frac{\vk}{k^2}.

where :math:`f` is the logarithmic growth rate, :math:`a` is the scale factor,
and :math:`H` is the Hubble parameter at :math:`a`. Note that bold variables
reflect vector quantities.

Finally, we Fourier transform :math:`\delta_L(\vk)` and :math:`\vv(\vk)` to
configuration space. These fields serve as the "initial conditions". They
will be evolved forward in time and Poisson sampled to create the final catalog.

**Perform the log-normal transformation**

Next, we perform a log-normal transformation on the density field :math:`\delta`.
As first discussed in `Coles and Jones 1991`_, the distribution of galaxies
on intermediate to large scales can be well-approximated by a log-normal
distribution. An additional useful property of a log-normal field is that
it follows the natural constraint :math:`\delta(\vx) \ge -1` of density
contrasts by definition. This property does not generally hold true for Gaussian
realizations of the overdensity field.

The new, transformed density field is given by

.. math::

    \delta(\vx) = e^{-\sigma^2 + b_L \delta_L(\vx)} - 1,

where the normalization factor :math:`\sigma^2` ensures that the mean of
the :math:`\delta(\vx)` vanishes and :math:`b_L` is the Lagrangian bias
factor, which is related to the final, linear bias as :math:`b_L = b_1 - 1`.
Here, :math:`b_1` is the value input by the user as the ``bias`` keyword
to the :class:`~lognormal.LogNormalCatalog` class.

**Poisson sample the density field**

We then generate discrete positions of objects by Poisson sampling the
overdensity field in each cell of the mesh. We assign each object the
velocity of the mesh cell that it is located in, and objects are placed randomly
inside their cells. The desired number density of objects in the box is
specified by the user as the ``nbar`` parameter to the
:class:`~lognormal.LogNormalCatalog` class.

**Apply the Zel'dovich approximation**

Finally, we evolve the overdensity field according to the
`Zel'dovich approximation <https://arxiv.org/abs/1401.5466>`_, which is 1st
order Lagrangian perturbation theory. To do this, we move the positions
of each object according to the linear velocity field,

.. math::

    \vr(\vx) = \vr_0(\vx) + \frac{\vv(\vx)}{f a H},

where :math:`\vr(\vx)` is the final position of the objects,
:math:`\vr_0(\vx)` is the initial position, and :math:`\vv(\vx)` is the
velocity assigned in the previous step.

After this step, we have a catalog of discrete objects, with a ``Position``
column in units of :math:`\mathrm{Mpc}/h`, a ``Velocity`` columns in units
of km/s, and a ``VelocityOffset`` column in units of :math:`\mathrm{Mpc}/h`.
The ``VelocityOffset`` is a convenience function for adding redshift-space
distortions (see :ref:`adding-rsd`), such that RSD can be added using:

.. code-block:: python

    line_of_sight = [0,0,1]
    src['Position'] = src['Position'] + src['VelocityOffset'] * line_of_sight

.. note::

  For examples using log-normal mocks, see the
  :ref:`cookbook/lognormal-mocks.ipynb` recipe in :ref:`cookbook`.

.. _hod-mock-data:

Halo Occupation Distribution Mocks
----------------------------------

nbodykit includes functionality to generate mock galaxy catalogs using the
`Halo Occupation Distribution`_ (HOD) technique via the :class:`~hod.HODCatalog`
class. The HOD technique populates a catalog of halos with galaxies based on
a functional form for the probability that a halo of mass :math:`M` hosts
:math:`N` objects, :math:`P(N|M)`. The functional form of the HOD used by
:class:`~hod.HODCatalog` is the form used in `Zheng et al 2007`_.
The average number of galaxies in a halo of mass :math:`M` is

.. math::

    \langle N_\mathrm{gal}(M) \rangle = N_\mathrm{cen}(M) \left [ 1 + N_\mathrm{sat}(M) \right],

where the occupation functions for centrals and satellites are given by

.. math::

    N_\mathrm{cen}(M) &= \frac{1}{2} \left (1  +  \mathrm{erf}
                \left[ \frac{\log_{10}M - \log_{10}M_\mathrm{min}}{\sigma_{\log_{10}M}}
                \right]
                \right),

    N_\mathrm{sat}(M) &= \left ( \frac{M - M_0}{M_1} \right )^\alpha.

This HOD parametrization has 5 parameters, which can be summarized as:

=============================== ============== =========== =======================================================================================
**Parameter**                   **Name**       **Default** **Description**
:math:`\log_{10}M_\mathrm{min}` ``logMmin``    13.031      Minimum mass required for a halo to host a central galaxy
:math:`\sigma_{\log_{10}M}`     ``sigma_logM`` 0.38        Rate of transition from :math:`N_\mathrm{cen}=0` to :math:`N_\mathrm{cen}=1`
:math:`\alpha`                  ``alpha``      0.76        Power law slope of the relation between halo mass and :math:`N_\mathrm{sat}`
:math:`\log_{10}M_0`            ``logM0``      13.27       Low-mass cutoff in :math:`N_\mathrm{sat}`
:math:`\log_{10}M_1`            ``logM1``      14.08       Characteristic halo mass where :math:`N_\mathrm{sat}` begins to assume a power law form
=============================== ============== =========== =======================================================================================

The default values of the HOD parameters are taken from
`Reid et al. 2014 <https://arxiv.org/abs/1404.3742>`_.

This form of the HOD clustering description assumes the galaxy -- halo connection
depends only on the halo mass. Thus, given a catalog of halo objects, with
associated mass values, users can quickly generate realistic galaxy catalogs
using this class.

.. _halo-catalog:

Interfacing with :mod:`halotools` via a :class:`~halos.HaloCatalog`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Internally, the :class:`~hod.HODCatalog` class uses the :mod:`halotools`
package to perform the halo population step. For further details,
see the documentation for :class:`halotools.empirical_models.Zheng07Cens` and
:class:`halotools.empirical_models.Zheng07Sats`, as well as
:ref:`this tutorial <zheng07_composite_model>` on the Zheng 07 HOD model.

The catalog of halos input to the :class:`~hod.HODCatalog` class must be of type
:class:`halotools.sim_manager.UserSuppliedHaloCatalog`, the tabular
data format preferred by :mod:`halotools`. nbodykit includes the
:class:`~halos.HaloCatalog` class in order to interface nicely with
:mod:`halotools`. In particular, this catalog object
includes a :func:`~halos.HaloCatalog.to_halotools` function to create a
:class:`~halotools.sim_manager.UserSuppliedHaloCatalog` from the data columns
in the :class:`~halos.HaloCatalog` object.

Given a :class:`CatalogSource` object, the :class:`~halos.HaloCatalog` object
interprets the objects as halos, using a specified redshift, cosmology,
and mass definition, to add several analytic columns to the catalog, including
:func:`~halos.HaloCatalog.Radius` and :func:`~halos.HaloCatalog.Concentration`.

For example, below we generate uniform particles in a box and then interpret
them as halos by specifying a redshift and cosmology:

.. ipython:: python

    from nbodykit.lab import HaloCatalog, cosmology

    # uniform objects in a box
    cat = UniformCatalog(nbar=100, BoxSize=1.0, seed=42)

    # add a Mass column to the objects
    cat['Mass'] = 10**(cat.rng.uniform(12, 15, size=cat.size))

    # initialize the halos
    halos = HaloCatalog(cat, cosmo=cosmology.Planck15, redshift=0., mdef='vir', position='Position', velocity='Velocity', mass='Mass')

    print(halos.columns)

And using the :func:`~halos.HaloCatalog.to_halotools` function, we can create
the :class:`halotools.sim_manager.UserSuppliedHaloCatalog` object needed
to initialize the :class:`~hod.HODCatalog` object.

.. ipython:: python

    halocat = halos.to_halotools()

    print(halocat)

    print(halocat.halo_table[:10])

**Caveats**

- The units of the halo position, velocity, and mass input to
  :class:`~halos.HaloCatalog` are assumed to be :math:`\mathrm{Mpc}/h`, km/s,
  and :math:`M_\odot/h`, respectively. These units are necessary to interface
  with :mod:`halotools`.
- The mass definition input to :class:`~halos.HaloCatalog` can be "vir"
  to use virial masses, or an overdensity factor with respect to the critical
  or mean density, i.e. "200c", "500c", or "200m", "500m".
- If using the built-in friends-of-friends (FOF) finder class,
  :class:`~nbodykit.algorithms.fof.FOF`, to identify halos, the user can use
  the :func:`~nbodykit.algorithms.fof.FOF.to_halos` function
  to directly produce a :class:`~halos.HaloCatalog` from the result of running
  the FOF algorithm.
- By default, the halo concentration values stored in the ``Concentration``
  column of a :class:`HaloCatalog` object are generated using the input mass
  definition and the analytic formulas from
  `Dutton and Maccio 2014`_. Users can
  overwrite this column with their own values if they wish to use custom
  concentration values when generating HOD catalogs.

The :class:`~hod.HODCatalog` Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can initialize the HOD catalog directly from the
:class:`~halotools.sim_manager.UserSuppliedHaloCatalog` object and the desired
HOD parameters. The :class:`~hod.HODCatalog` object will include all of the
columns from the :class:`~halotools.sim_manager.UserSuppliedHaloCatalog` object,
with the usual columns ``Position``, ``Velocity``, and ``VelocityOffset``
for the generated galaxies. The additional columns are:

#. **conc_NFWmodel**: the concentration of the halo
#. **gal_type**: the galaxy type, 0 for centrals and 1 for satellites
#. **halo_id**: the global ID of the halo that this galaxy belongs to,
   between 0 and :attr:`csize`
#. **halo_local_id**: the local ID of the halo that this galaxy belongs to,
   between 0 and :attr:`size`
#. **halo_mvir**: the halo mass, in units of :math:`M_\odot/h`
#. **halo_nfw_conc**: alias of ``conc_NFWmodel``
#. **halo_num_centrals**: the number of centrals that this halo hosts,
   either 0 or 1
#. **halo_num_satellites**: the number of satellites that this halo hosts
#. **halo_rvir**: the halo radius, in units of :math:`\mathrm{Mpc}/h`
#. **halo_upid**: equal to -1; should be ignored by the user
#. **halo_vx, halo_vy, halo_vz**: the three components of the halo velocity,
   in units of km/s
#. **halo_x, halo_y, halo_z**: the three components of the halo position,
   in units of :math:`\mathrm{Mpc}/h`
#. **host_centric_distance**: the distance from this galaxy to the center of
   the halo, in units of :math:`\mathrm{Mpc}/h`
#. **vx, vy, vz**: the three components of the galaxy velocity, equal to
   ``Velocity``, in units of km/s
#. **x,y,z**: the three components of the galaxy position, equal to
   ``Position``, in units of :math:`\mathrm{Mpc}/h`

Below we initialize a :class:`~hod.HODCatalog` of galaxies and compute the
number of centrals and satellites in the catalog, using the ``gal_type``
column.

.. ipython:: python

    from nbodykit.lab import HODCatalog
    hod = HODCatalog(halocat, alpha=0.5, sigma_logM=0.40, seed=42)

    print("total number of HOD galaxies = ", hod.csize)
    print(hod.columns)

    print("number of centrals = ", hod.compute((hod['gal_type']==0).sum()))
    print("number of satellites = ", hod.compute((hod['gal_type']==1).sum()))

**Caveats**

- The HOD population step requires halo concentration. If the user wishes to
  uses custom concentration values, the
  :class:`~halotools.sim_manager.UserSuppliedHaloCatalog` table should contain
  a ``halo_nfw_conc`` column. Otherwise, the analytic prescriptions from
  `Dutton and Maccio 2014`_ are used.

Repopulating a HOD Catalog
^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also quickly repopulate a HOD catalog in place, generating a new
set of galaxies for the same set of halos, either changing the random seed
or the HOD parameters. For example,

.. ipython:: python

    # repopulate, just changing the random seed
    hod.repopulate(seed=84)
    print("total number of HOD galaxies = ", hod.csize)

    print("number of centrals = ", hod.compute((hod['gal_type']==0).sum()))
    print("number of satellites = ", hod.compute((hod['gal_type']==1).sum()))

    # re-populate with new parameters
    hod.repopulate(logM0=13.2, logM1=14.5)
    print("total number of HOD galaxies = ", hod.csize)

    print("number of centrals = ", hod.compute((hod['gal_type']==0).sum()))
    print("number of satellites = ", hod.compute((hod['gal_type']==1).sum()))

.. note::

  For examples using HOD mocks, see the :ref:`cookbook/hod-mocks.ipynb`
  recipe in :ref:`cookbook`.

.. _custom-hod-mocks:

Using a Custom HOD Model
^^^^^^^^^^^^^^^^^^^^^^^^

Users can implement catalogs that use custom HOD modeling by subclassing
the :class:`~hod.HODBase` class. This base class is abstract, and subclasses must
implement the :func:`~hod.HODBase.__makemodel__` function. This function
returns a :class:`~halotools.empirical_models.HodModelFactory`
object, which is the :mod:`halotools` object responsible for supporting
custom HOD models. For more information on designing your own HOD model
using :mod:`halotools`, see
:ref:`this series of halotools tutorials <hod_modeling_tutorial0>`.

.. _Coles and Jones 1991: http://adsabs.harvard.edu/abs/1991MNRAS.248....1C
.. _Halo Occupation Distribution: https://arxiv.org/abs/astro-ph/0212357
.. _Zheng et al 2007: https://arxiv.org/abs/astro-ph/0703457
.. _Dutton and Maccio 2014: https://arxiv.org/abs/1402.7073
