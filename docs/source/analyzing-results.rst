.. currentmodule:: nbodykit.binned_statistic

.. _analyzing-results:

Analyzing your Results
======================

Several nbodykit algorithms compute binned clustering statistics
and store the results as a :class:`BinnedStatistic` object (see a list
of these algorithms :ref:`here <api-clustering-statistics>`. In this section,
we provide an overview of some of the functionality of this class to help
users better analyze their algorithm results.

The :class:`BinnedStatistic` class is designed to hold data variables at
fixed coordinates, i.e., a grid of :math:`(r, \mu)` or :math:`(k, \mu)` bins
and is modeled after the syntax of the :class:`xarray.Dataset` object.

Loading and Saving Results
--------------------------

A :class:`BinnedStatistic` object is serialized to disk using a
JSON format. The :func:`~BinnedStatistic.to_json` and
:func:`~BinnedStatistic.from_json` functions can be used to save and load
:class:`BinnedStatistic` objects. respectively.

To start, we read two :class:`BinnedStatistic` results from JSON files,
one holding 1D power measurement :math:`P(k)` and one holding a 2D power
measurement :math:`P(k,\mu)`.

.. ipython:: python
  :suppress:

  import os
  from nbodykit import style
  import matplotlib.pyplot as plt
  plt.style.use(style.notebook)

  data_dir = os.path.join(os.environ['SOURCE_DIR'], 'data')

.. ipython:: python


  from nbodykit.binned_statistic import BinnedStatistic
  power_1d = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_1d.json'))
  power_2d = BinnedStatistic.from_json(os.path.join(data_dir, 'dataset_2d.json'))


Coordinate Grid
---------------

The clustering statistics are measured for fixed bins, and the
:class:`BinnedStatistic` class has several attributes to access the
coordinate grid defined by these bins:

- :attr:`shape`: the shape of the coordinate grid
- :attr:`dims`: the names of each dimension of the coordinate grid
- :attr:`coords`: a dictionary that gives the center bin values for each dimension of the grid
- :attr:`edges`: a dictionary giving the edges of the bins for each coordinate dimension

.. ipython:: python

    print(power_1d.shape, power_2d.shape)

    print(power_1d.dims, power_2d.dims)

    power_2d.coords

    power_2d.edges

Accessing the Data
------------------

The names of data variables stored in a :class:`BinnedStatistic` are stored in
the :attr:`variables` attribute, and the :attr:`data` attribute stores the
arrays for each of these names in a structured array. The data for a given
variable can be accessed in a dict-like fashion:

.. ipython:: python

    power_1d.variables
    power_2d.variables

    # the real component of the 1D power
    Pk = power_1d['power'].real
    print(type(Pk), Pk.shape, Pk.dtype)

    # complex power array
    Pkmu = power_2d['power']
    print(type(Pkmu), Pkmu.shape, Pkmu.dtype)

In some cases, the variable value for a given bin will be missing or invalid,
which is indicated by a :data:`numpy.nan` value in the :attr:`data` array for
the given bin. The :class:`BinnedStatistic` class carries a :attr:`mask`
attribute that defines which elements of the data array have
:data:`numpy.nan` values.

Meta-data
---------

An :class:`~collections.OrderedDict` of meta-data for a
:class:`BinnedStatistic` class is stored in the :attr:`attrs` attribute.
Typically in nbodykit, the :attr:`attrs` dictionary stores
information about box size, number of objects, etc:

.. ipython:: python

    power_2d.attrs

To attach additional meta-data to a :class:`BinnedStatistic` class, the user
can add additional keywords to the :attr:`attrs` dictionary.

Slicing
-------

Slices of the coordinate grid of a :class:`BinnedStatistic` can be achieved
using array-like indexing of the main :class:`BinnedStatistic` object, which
will return a new :class:`BinnedStatistic` holding the sliced data:

.. ipython:: python

    # select the first mu bin
    power_2d[:,0]

    # select the first and last mu bins
    power_2d[:, [0, -1]]

    # select the first 5 k bins
    power_1d[:5]

A typical usage of array-like indexing is to loop over the ``mu`` dimension
of a 2D :class:`BinnedStatistic`, such as when plotting:

.. ipython:: python
    :okwarning:

    from matplotlib import pyplot as plt

    # the shot noise is volume / number of objects
    shot_noise = power_2d.attrs['volume'] / power_2d.attrs['N1']

    # plot each mu bin separately
    for i in range(power_2d.shape[1]):
        pk = power_2d[:,i]
        label = r"$\mu = %.1f$" % power_2d.coords['mu'][i]
        plt.loglog(pk['k'], pk['power'].real - shot_noise, label=label)

    plt.legend()
    plt.xlabel(r"$k$ [$h$/Mpc]", fontsize=14)
    plt.ylabel(r"$P(k,\mu)$ $[\mathrm{Mpc}/h]^3$", fontsize=14)

    @savefig BinnedStatistic_pkmu_plot.png width=6in
    plt.show()

The coordinate grid can also be sliced using label-based indexing, similar to
the syntax of :meth:`xarray.Dataset.sel`. The ``method`` keyword of
:func:`~BinnedStatistic.sel` determines if exact
coordinate matching is required (``method=None``, the default) or if the
nearest grid coordinate should be selected automatically (``method='nearest'``).

For example, we can slice power spectrum results based on the
``k`` and ``mu`` coordinate values:

.. ipython:: python

    # get all mu bins for the k bin closest to k=0.1
    power_2d.sel(k=0.1, method='nearest')

    # slice from k=0.01-0.1 for mu = 0.5
    power_2d.sel(k=slice(0.01, 0.1), mu=0.5, method='nearest')

We also provide a :func:`~BinnedStatistic.squeeze` function which behaves
similar to the :func:`numpy.squeeze` function:

.. ipython:: python

    # get all mu bins for the k bin closest to k=0.1, but keep k dimension
    sliced = power_2d.sel(k=[0.1], method='nearest')
    sliced

    # and then squeeze to remove the k dimension
    sliced.squeeze()

Note that, by default, array-based or label-based indexing will
automatically "squeeze" sliced objects that have a dimension of length one,
unless a list of indexers is used, as is done above.

Reindexing
----------

It is possible to reindex a specific dimension of the coordinate grid using
:func:`~BinnedStatistic.reindex`. The new bin spacing must be an integral
multiple of the original spacing, and the variable values will be averaged together
on the new coordinate grid.

.. ipython:: python
    :okwarning:

    power_2d.reindex('k', 0.02)

    power_2d.reindex('mu', 0.4)

.. note::
    Any variable names passed to :func:`~BinnedStatistic.reindex` via the
    ``fields_to_sum`` keyword will have their values summed, instead of averaged,
    when reindexing.

Averaging
---------

The average of a specific dimension can be taken using
the :func:`~BinnedStatistic.average` function. A common use case is averaging
over the ``mu`` dimension of a 2D :class:`BinnedStatistic`, which is accomplished
by:

.. ipython:: python

    # compute P(k) from P(k,mu)
    power_2d.average('mu')
