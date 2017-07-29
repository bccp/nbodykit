.. _mock-catalogs:

.. currentmodule:: nbodykit.source.catalog

Generating Catalogs of Mock Data
================================

nbodykit includes several methods for generating mock catalogs, with varying
levels of sophistication. These :class:`~nbodykit.base.catalog.CatalogSource`
objects allow users to create catalogs of objects at run time and include:

* :ref:`random-mock-data`
* :ref:`lognormal-mock-data`
* :ref:`hod-mock-data`

.. _random-mock-data:

Randomly Distributed Objects
----------------------------

nbodykit :class:`~nbodykit.base.catalogs.CatalogSource` subclasses that
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
of MPI ranks being used. The random number generator is stored as the
:attr:`~uniform.RandomCatalog.rng` attribute. Users can use this random
number generator to add columns to the catalog, using the :ref:`syntax to add
columns <adding-columns>`.

For example,

.. ipython:: python

    from nbodykit.lab import RandomCatalog
    import numpy

    # initialize the catalog with no columns yet
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

**Generating Gaussian initial conditions**

First, a Gaussian overdensity field :math:`\delta_L(k)` is generated in
Fourier space with a power spectrum given by the input function specified by
the user. We also generate linear velocity fields from the overdensity field
in the :math:`i^\mathrm{th}` direction in Fourier space, using

.. math::

    v_i(k) = \frac{i f a H }{k^2} k_i \delta_L(k).

where :math:`f` is the logarithmic growth rate, :math:`a` is the scale factor,
and :math:`H` is the Hubble parameter at :math:`a`.

Finally, we Fourier transform :math:`\delta_L(k)` and :math:`v(k)` to
configuration space. These fields serve as the "initial conditions". They
will be evolved forward in time and Poisson-sampled to create the final catalog.

**Performing the log-normal transformation**

Next, we perform a log-normal transformation on the density field :math:`\delta`.
As first discussed in `Coles and Jones 1991`_, the distribution of galaxies
on intermediate to large scales can be well-approximated by a log-normal
distribution. An additional nice property of a log-normal field is that
it follows the natural constraint :math:`\delta(x) \ge -1` of density
contrasts by definition. This property does not hold true for Gaussian
realizations of the overdensity field.

The new, transformed density field is given by

.. math::

    \delta(x) = e^{-\sigma^2 + b_L \delta_L(x)} - 1,

where the normalization factor :math:`\sigma^2` ensures that the mean of
the :math:`\delta(x)` vanishes and :math:`b_L` is the Lagrangian bias
factor, which is related to the final linear bias as :math:`b_L = b_1 - 1`.
Here, :math:`b_1` is the value input by the user as the ``bias`` keyword
to the :class:`~lognormal.LogNormalCatalog` class.

**Poisson-sampling the density field**

We then generate discrete positions of objects by Poisson sampling the
overdensity field in each cell of the mesh. We assign each object the
velocity of the mesh cell that it is located in, and objects are placed randomly
inside their cells. The desired number density of objects in the box is specified
by the user as the ``nbar`` parameter to the
:class:`~lognormal.LogNormalCatalog` class.

**Applying the Zel'dovich approximation**

Finally, we evolve the overdensity field according to the
`Zel'dovich approximation <https://arxiv.org/abs/1401.5466>`_, which is 1st
order Lagrangian perturbation theory. To do this, we move the positions
of each object according to the linear velocity field,

.. math::

    r(x) = r_0(x) + \frac{v(x)}{f a H},

where :math:`r(x)` is the final position of the objects, :math:`r_0(x)` is the
initial position, and :math:`v(x)` is the velocity assigned
in the previous step.

After this step, we have a catalog of discrete objects, with a ``Position``
column in units of :math:`\mathrm{Mpc}/h`, a ``Velocity`` columns in units
of km/s, and a ``VelocityOffset`` column in units of :math:`\mathrm{Mpc}/h`.
The ``VelocityOffset`` is a convenience function for adding redshift-space
distortions (see :ref:`adding-rsd`), such that RSD can be added using:

.. code-block:: python

    line_of_sight = [0,0,1]
    src['Position'] = src['Position'] + src['VelocityOffset'] * line_of_sight

.. note::

  For worked examples using log-normal mocks, see the :ref:`cookbook/lognormal.ipynb`
  of :ref:`cookbook`.

.. _hod-mock-data:

Halo Occupation Distribution Mocks
----------------------------------


.. _Cole and Jones 1991: <http://adsabs.harvard.edu/abs/1991MNRAS.248....1C>
