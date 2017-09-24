.. currentmodule:: nbodykit.base.catalog

.. ipython:: python
    :suppress:

    import tempfile, os
    startdir = os.path.abspath('.')
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    numpy.random.seed(42)


.. _common-operations:

Common Data Operations
======================

Here, we detail some of the most common operations when dealing with
data in the form of a :class:`CatalogSource`. The native format for data columns
in a :class:`CatalogSource` object is the dask array. Be sure to read
the :ref:`previous section <on-demand-io>` for an introduction to dask arrays
before proceeding.

The dask array format allows users to easily
manipulate columns in their input data and feed any transformed data into one
of the nbodykit algorithms. This provides a fast and easy way to transform
the data while hiding the implementation details
needed to compute these transformations internally. In this section, we'll
provide examples of some of these data transformations to get users
acclimated to dask arrays quickly.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

To help illustrate these operations, we'll initialize the nbodykit "lab"
and load a catalog of uniformly distributed objects.

.. ipython:: python

  from nbodykit.lab import *
  cat = UniformCatalog(nbar=100, BoxSize=1.0, seed=42)

.. _accessing-columns:

Accessing Data Columns
-----------------------

Specific columns can be accessed by indexing the catalog object using the
column name, and a :class:`dask.array.Array` object is returned (see
:ref:`what-is-dask-array` for more details on dask arrays).

.. ipython:: python

  position = cat['Position']
  velocity = cat['Velocity']

  print(position)
  print(velocity)

While in the format of the dask array, data columns can easily be manipulated
by the user. For example, here we normalize the position coordinates to the
range 0 to 1 by dividing by the box size:

.. ipython:: python

  # normalize the position
  normed_position = position / cat.attrs['BoxSize']

  print(normed_position)

Note that the normalized position array is also a dask array and that the
actual normalization operation is yet to occur. This makes these kinds of
data transformations very fast for the user.

Computing Data Columns
----------------------

Columns can be converted from :class:`dask.array.Array` objects to
numpy arrays using the :func:`~CatalogSource.compute` function (see
:ref:`evaluating-dask-array` for further details on computing dask arrays).

.. ipython:: python

  position, velocity = cat.compute(cat['Position'], cat['Velocity'])

  print(type(position))
  print(type(velocity))

We can also compute the max of the normalized position coordinates from
the previous section:

.. ipython:: python

  maxpos = normed_position.max(axis=0)
  print(maxpos)

  print(cat.compute(maxpos))

.. _adding-columns:

Adding New Columns
------------------

New columns can be easily added to a :class:`CatalogSource` object by
directly setting them:

.. ipython:: python

  # no "Mass" column originally
  print("contains 'Mass'? :", 'Mass' in cat)

  # add a random array as the "Mass" column
  cat['Mass'] = numpy.random.random(size=len(cat))

  # "Mass" exists!
  print("contains 'Mass'? :", 'Mass' in cat)

  # can also add scalar values -- converted to correct length
  cat['Type'] = b"central"

  print(cat['Mass'])
  print(cat['Type'])

Here, we have added two new columns to the catalog, ``Mass`` and ``Type``.
Internally, nbodykit stores the new columns as dask arrays and
will automatically convert them to the correct type if they are not already.

**Caveats**

- New columns must be either be a scalar value, or an array with the same
  length as the catalog. Scalar values will automatically be broadcast to
  the correct length.
- Setting a column of the wrong length will raise an exception.

.. _overwriting-columns:

Overwriting Columns
-------------------

The same syntax used for adding new columns can also be used to overwrite
columns that already exist in a :class:`CatalogSource`. This procedure
works as one would expect -- the most up-to-date values of columns are
always used in operations.

In the example below we overwrite both the ``Position`` and ``Velocity``
columns, and each time the columns are accessed, the most up-to-date values
are used, as expected.

.. ipython:: python

  # some fake data
  data = numpy.ones(5, dtype=[
          ('Position', ('f4', 3)),
          ('Velocity', ('f4', 3))]
          )

  # initialize a catalog directly from the structured array
  src = ArrayCatalog(data)

  # overwrite the Velocity column
  src['Velocity'] = src['Position'] + src['Velocity'] # 1 + 1 = 2

  # overwrite the Position column
  src['Position'] = src['Position'] + src['Velocity'] # 1 + 2 = 3

  print("Velocity = ", src.compute(src['Velocity'])) # all equal to 2
  print("Position = ", src.compute(src['Position'])) # all equal to 3

.. _adding-rsd:

Adding Redshift-space Distortions
---------------------------------

A useful operation in large-scale structure is the mapping of positions
in simulations from real space to redshift space, referred to
as `redshift space distortions <https://arxiv.org/abs/astro-ph/9708102>`_ (RSD).
This operation can be easily performed by combining the ``Position`` and
``Velocity`` columns to overwrite the ``Position`` column. As first
found by `Kaiser 1987 <http://adsabs.harvard.edu/abs/1987MNRAS.227....1K>`_,
the mapping from real to redshift space is:

.. math::

    s = r + \frac{\vv \cdot \nhat}}{a H},

where :math:`r` is the line-of-sight position in real space,
:math:`s` is the line-of-sight position in redshift space, :math:`\vv` is the
velocity vector, :math:`\vnhat` is the line-of-sight unit vector, :math:`a` is
the scale factor, and :math:`H` is the Hubble parameter at :math:`a`.

As an example, below we add RSD along the ``z`` axis of a simulation box:

.. ipython:: python

    # apply RSD along the z axis
    line_of_sight = [0,0,1]

    # redshift and cosmology
    redshift =  0.55; cosmo = cosmology.Cosmology(Om0=0.31, H0=70)

    # the RSD normalization factor
    rsd_factor = (1+redshift) / (100 * cosmo.efunc(redshift))

    # update Position, applying RSD
    src['Position'] = src['Position'] + rsd_factor * src['Velocity'] * line_of_sight

The RSD factor is known as the conformal Hubble parameter
:math:`\mathcal{H} = a H(a)`. This calculation requires a cosmology,
which can be specified via the
:class:`~nbodykit.cosmology.core.Cosmology` class. We use the
:func:`~nbodykit.cosmology.core.Cosmology.efunc` function which returns
:math:`E(z)`, where the Hubble parameter is defined as :math:`H(z) = 100h\ E(z)`
in units of km/s/Mpc. Note that the operation above assumes the ``Position``
column is in units of :math:`\mathrm{Mpc}/h`.

For catalogs in nbodykit that generate mock data, such as the
:ref:`log-normal catalogs <lognormal-mock-data>` or :ref:`HOD catalogs <hod-mock-data>`,
there is an additional column, ``VelocityOffset``, available to facilitate
RSD operations. This column has units of :math:`\mathrm{Mpc}/h` and
includes the ``rsd_factor`` above. Thus, this allows users to add RSD
simply by using:

.. code-block:: python

    src['Position'] = src['Position'] + src['VelocityOffset'] * line_of_sight


Selecting a Subset
------------------

A subset of a :class:`CatalogSource` object can be selected using slice notation.
There are two ways to select a subset:

#. use a boolean array, which specifies which rows of the catalog to select
#. use a slice object specifying which rows of the catalog to select

For example,

.. ipython:: python

    # boolean selection array
    select = cat['Mass'] < 0.5
    print("number of True entries = ", cat.compute(select.sum()))

    # select only entries where select = True
    subcat = cat[select]

    print("size of subcat = ", subcat.size)

    # select the first ten rows
    subcat = cat[:10]
    print("size of subcat = ", subcat.size)

    # select first and last row
    subcat = cat[[0, -1]]
    print("size of subcat = ", subcat.size)

**Caveats**

- When indexing with a boolean array, the array must have the same length as
  the :attr:`size` attribute, or an exception will be raised.
- Selecting a single object by indexing with an integer is not supported. If
  the user wishes to select a single row, a list of length one can be used to
  select the specific row.

Selecting a Subset of Columns from a ``CatalogSource``
------------------------------------------------------

A subset of columns can be selected from a :class:`CatalogSource` object by
indexing the catalog with a list of the names of the desired columns.
For example,

.. ipython:: python

    print("columns in catalog = ", cat.columns)

    # select Position + Mass
    subcat = cat[['Position', 'Mass']]

    # the selected columns + default columns
    print("columns in subset = ", subcat.columns)

**Caveats**

- When selecting a subset of columns, note that in addition to the desired columns,
  the sub-catalog will also contain the
  :ref:`default columns <catalog-source-default-columns>` (``Weight``, ``Value``,
  and ``Selection``).

.. _transform-ops:

The ``nbodykit.transform`` module
---------------------------------

The :mod:`nbodykit.transform` module includes several commonly used functions
for convenience. We describe a few of the most common use cases in the sub-sections
below.

.. note::
  The :mod:`~nbodykit.transform` module is available to users when
  ``from nbodykit.lab import *`` is executed.

.. _combining-sources:

Concatenating ``CatalogSource`` Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :class:`CatalogSource` objects have the same columns, they can be
concatenated together into a single object using the
:func:`nbodykit.transform.ConcatenateSources` function. For example,

.. ipython:: python

    cat1 = UniformCatalog(nbar=50, BoxSize=1.0, seed=42)
    cat2 = UniformCatalog(nbar=150, BoxSize=1.0, seed=42)

    combined = transform.ConcatenateSources(cat1, cat2)

    print("total size = %d + %d = %d" %(cat1.size, cat2.size, combined.size))


.. _stacking-columns:

Stacking Columns Together
^^^^^^^^^^^^^^^^^^^^^^^^^

Another common use case is when users need to combine separate
data columns vertically, as the columns of a new array. For example, often the
Cartesian position coordinates ``x``, ``y``, and ``z`` are stored as separate
columns, and the ``Position`` column must be added to a catalog from these
individual columns. We provide the :func:`nbodykit.transform.StackColumns`
function for this exact purpose. For example,

.. ipython:: python

    # fake position data
    data = numpy.random.random(size=(5,3))

    # save to a plaintext file
    numpy.savetxt('csv-example.dat', data, fmt='%.7e')

    # the cartesian coordinates
    names =['x', 'y', 'z']

    # read the data
    f = CSVCatalog('csv-example.dat', names)

    # make the "Position" column
    f['Position'] =  transform.StackColumns(f['x'], f['y'], f['z'])

    print(f['Position'])
    print(f.compute(f['Position']))

.. _sky-to-cartesian:

Converting from Sky to Cartesian Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide the function :func:`nbodykit.transform.SkyToCartesian` for converting
sky coordinates, in the form of right ascension, declination, and redshift,
to Cartesian coordinates. The conversion from redshift to comoving distance
requires a cosmology instance, which can be specified via the
:class:`~nbodykit.cosmology.core.Cosmology` class.

Below, we initialize a catalog holding random right ascension,
declination, and redshift coordinates, and then add the Cartesian position
as the ``Position`` column.

.. ipython:: python

    src = RandomCatalog(100, seed=42)

    # add random (ra, dec, z) coordinates
    src['z'] = src.rng.normal(loc=0.5, scale=0.1, size=src.size)
    src['ra'] = src.rng.uniform(low=0, high=360, size=src.size)
    src['dec'] = src.rng.uniform(low=-180, high=180., size=src.size)

    # initialize a set of cosmology parameters
    cosmo = cosmology.Cosmology(Om0=0.31, H0=70)

    # add the position
    src['Position'] = transform.SkyToCartesian(src['ra'], src['dec'], src['z'], degrees=True, cosmo=cosmo)

**Caveats**

- Whether the right ascension and declination arrays are in degrees
  (as opposed to radians) should be specified via the ``degrees`` keyword.
- The units of the returned ``Position`` column are :math:`\mathrm{Mpc}/h`.

.. _transform-da:

Using the ``dask.array`` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more general column transformations, users should take advantage of the
:mod:`dask.array` module, which implements most functions in the :mod:`numpy`
package in a manner optimized for dask arrays. The module can be accessed from the
:mod:`nbodykit.transform` module as :mod:`nbodykit.transform.da`.


.. important::

    For a full list of functions available in the :mod:`dask.array` module,
    please see the :doc:`dask array documentation <dask:array-api>`.
    We strongly recommend that new users read through this documentation
    and familiarize themselves with the functionality provided by
    the :mod:`dask.array` module.

As a simple illustration, below we convert an array holding right ascension values from
degrees to radians, compute the sine of the array, and find the min and
max values using functions available in the :mod:`dask.array` module.

.. ipython:: python

    ra = transform.da.deg2rad(src['ra']) # from degrees to radians
    sin_ra = transform.da.sin(ra) # compute the sine

    print("min(sin(ra)) = ", src.compute(sin_ra.min()))
    print("max(sin(ra)) = ", src.compute(sin_ra.max()))

.. ipython:: python
    :suppress:

    import shutil
    os.chdir(startdir)
    shutil.rmtree(tmpdir)
