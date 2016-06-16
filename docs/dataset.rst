DataSet for Algorithm results
=============================

Several nbodykit algorithms compute two-point clustering statistics, 
and we provide the :class:`~nbodykit.dataset.DataSet` class for analyzing these results. 
The class is designed to hold data variables at fixed coordinates, 
i.e., a grid of :math:`(r, \mu)` or :math:`(k, \mu)` bins.
    
The DataSet class is modeled after the syntax of :class:`xarray.Dataset`, and there
are several subclasses of DataSet that are specifically designed to hold correlation 
function or power spectrum results (in 1D or 2D).

For algorithms that compute power spectra, we have:

* :class:`FFTPower <nbodykit.plugins.algorithms.PeriodicBox.FFTPowerAlgorithm>`
    - computes: :math:`P(k, \mu)` or :math:`P(k)`
    - results class: :class:`nbodykit.dataset.Power2dDataSet` or :class:`nbodykit.dataset.Power1dDataSet`
* :class:`BianchiFFTPower <nbodykit.plugins.algorithms.BianchiFFTPower.BianchiFFTPowerAlgorithm>`
    - computes: :math:`P(k)`
    - results class: :class:`nbodykit.dataset.Power1dDataSet`
    
And for algorithms computing correlation functions:

* :class:`FFTCorrelation <nbodykit.plugins.algorithms.PeriodicBox.FFTCorrelationAlgorithm>`, :class:`PairCountCorrelation <nbodykit.plugins.algorithms.PairCountCorrelation.PairCountCorrelationAlgorithm>`
    - computes: :math:`\xi(k, \mu)` or :math:`\xi(k)`
    - results class: :class:`nbodykit.dataset.Corr2dDataSet` or :class:`nbodykit.dataset.Corr1dDataSet`

.. ipython:: python
   :suppress:
   
   from __future__ import print_function
   from nbodykit.test import download_results_file, cache_dir
   import os
   
   targetdir = os.path.join(cache_dir, 'results')
   download_results_file('test_power_plaintext.dat', targetdir) # 2d result
   download_results_file('test_power_cross.dat', targetdir) # 1d result
      
Loading results
---------------

To load power spectrum or correlation function results, the user must first read the plaintext
files and then initialize the relevant subclass of DataSet. The functions :func:`nbodykit.files.Read2DPlainText`
and :func:`nbodykit.files.Read1DPlainText` should be used for reading 2D and 1D result files, respectively.

The reading and DataSet initialization can be performed in one step, taking advantage of
:func:`~nbodykit.dataset.DataSet.from_nbkit`:
    
.. ipython:: python

    from nbodykit import dataset, files
    
    # output file of 'examples/power/test_plaintext.params'
    filename_2d = os.path.join(cache_dir, 'results', 'test_power_plaintext.dat')
    
    # load a 2D power result
    power_2d =  dataset.Power2dDataSet.from_nbkit(*files.Read2DPlainText(filename_2d))
    power_2d
    
    # output file of 'examples/power/test_cross_power.params'
    filename_1d =  os.path.join(cache_dir, 'results', 'test_power_cross.dat')
    
    # load a 1D power result
    power_1d =  dataset.Power1dDataSet.from_nbkit(*files.Read1DPlainText(filename_1d))
    power_1d
    
Coordinate grid
---------------

The clustering statistics are measured for fixed bins, and the DataSet class
has several attributes to access the coordinate grid defined by these bins:


    - :attr:`shape`: the shape of the coordinate grid
    - :attr:`dims`: the names of each dimension of the coordinate grid
    - :attr:`coords`: a dictionary that gives the center bin values for each dimension of the grid
    - :attr:`edges`: a dictionary giving the edges of the bins for each coordinate dimension

.. ipython:: python
    
    print(power_1d.shape, power_2d.shape)
    
    print(power_1d.dims, power_2d.dims)

    power_2d.coords
    
    power_2d.edges
    
The center bin values can also be directly accessed in a dict-like fashion 
from the main DataSet using the dimension names:

.. ipython :: python

    power_2d['k_cen'] is power_2d.coords['k_cen']
    power_2d['mu_cen'] is power_2d.coords['mu_cen']


Accessing the data
------------------

The names of data variables stored in a DataSet are stored in the :attr:`variables` attribute,
and the :attr:`data` attribute stores the arrays for each of these names in a structured array. The
data for a given variable can be accessed in a dict-like fashion:

.. ipython:: python
    
    power_1d.variables
    power_2d.variables

    # the real component of the power
    Pk = power_1d['power.real']
    print(type(Pk), Pk.shape, Pk.dtype)
    
    
    # complex power array
    Pkmu = power_2d['power']
    print(type(Pkmu), Pkmu.shape, Pkmu.dtype)
    
In some cases, the variable value for a given bin will be missing or invalid, which is 
indicated by a :data:`numpy.nan` value in the :attr:`data` array for the given bin. 
The DataSet class carries a :attr:`mask` attribute that defines which elements
of the data array have :data:`numpy.nan` values.
    
Meta-data
---------

An :class:`~collections.OrderedDict` of meta-data for a DataSet class is 
stored in the :attr:`attrs` attribute. The :func:`~nbodykit.files.Read1DPlainText` 
and :func:`~nbodykit.files.Read2DPlainText` functions will load any meta-data 
saved to file while running an algorithm. 

Typically for power spectrum and correlation function results, the 
:attr:`attrs` dictionary stores information about box size, number of objects, etc: 

.. ipython:: python

    power_2d.attrs

To attach additional meta-data to a DataSet class, the user can add additional
keywords to the :attr:`attrs` dictionary.

Slicing
-------

Slices of the coordinate grid of a DataSet can be achieved using array-like indexing 
of the main DataSet class, which will return a new DataSet holding the sliced data:

.. ipython:: python

    # select the first mu bin
    power_2d[:,0]
    
    # select the first and last mu bins
    power_2d[:, [0, -1]]
    
    # select the first 5 k bins
    power_1d[:5]
    
A typical usage of array-like indexing is to loop over the `mu_cen` dimension 
of a 2D DataSet, such as when plotting: 

.. ipython:: python
    :okwarning:

    from matplotlib import pyplot as plt

    # the shot noise is volume / number of objects
    shot_noise = power_2d.attrs['volume'] / power_2d.attrs['N1']
    
    # plot each mu bin separately 
    for i in range(power_2d.shape[1]):
        pk = power_2d[:,i]
        label = r"$\mu = %.1f$" % power_2d['mu_cen'][i] 
        plt.loglog(pk['k'], pk['power'].real - shot_noise, label=label)
        
    print(os.getcwd())
    
    plt.legend()
    plt.xlabel(r"$k$ [$h$/Mpc]", fontsize=14)
    plt.ylabel(r"$P(k,\mu)$ $[\mathrm{Mpc}/h]^3$", fontsize=14)    
    
    @savefig dataset_pkmu_plot.png width=6in
    plt.show()
    
The coordinate grid can also be sliced using label-based indexing, similar to 
the syntax of :meth:`xarray.Dataset.sel`. The ``method`` keyword of 
:func:`~nbodykit.dataset.DataSet.sel` determines if exact coordinate matching
is required (``method=None``, the default) or if the nearest grid coordinate
should be selected automatically (``method='nearest'``).

For example, we can slice power spectrum results based on `k_cen` and `mu_cen`
values:

.. ipython:: python 

    # get all mu bins for the k bin closest to k=0.1
    power_2d.sel(k_cen=0.1, method='nearest')
    
    # slice from k=0.01-0.1 for mu = 0.5
    power_2d.sel(k_cen=slice(0.01, 0.1), mu_cen=0.5, method='nearest')

We also provide a function :func:`~nbodykit.dataset.DataSet.squeeze` with functionality
similar to :func:`numpy.squeeze` for the DataSet class:


.. ipython:: python

    # get all mu bins for the k bin closest to k=0.1, but keep k dimension
    sliced = power_2d.sel(k_cen=[0.1], method='nearest')
    sliced
    
    # and then squeeze to remove the k dimension
    sliced.squeeze()
    
Note that, by default, array-based or label-based indexing will automatically "squeeze"
sliced objects that have a dimension of length one, unless a list of indexers is used, as is done above. 

Reindexing
----------

It is possible to reindex a specific dimension of the coordinate grid using 
:func:`~nbodykit.dataset.DataSet.reindex`. The new bin spacing must be an integral 
multiple of the original spacing, and the variable values will be averaged together
on the new coordinate grid. 

.. ipython:: python
    :okwarning:

    power_2d.reindex('k_cen', 0.02)
    
    power_2d.reindex('mu_cen', 0.4)
    
Any variable names passed to :func:`~nbodykit.dataset.DataSet.reindex` via the `fields_to_sum`
keyword will have their values summed, instead of averaged, when reindexing. Futhermore, 
for :class:`~nbodykit.dataset.Power2dDataSet` and :class:`~nbodykit.dataset.Power1dDataSet`, 
the ``modes`` variable will be automatically summed, and for 
:class:`~nbodykit.dataset.Corr2dDataSet` or :class:`~nbodykit.dataset.Corr1dDataSet`, 
the ``N`` and ``RR`` fields will be automatically summed when reindexing.
    
Averaging
---------

The average of a specific dimension can be taken using :func:`~nbodykit.dataset.DataSet.average`.
A common usage is averaging over the `mu_cen` dimension of a 2D DataSet, which is accomplished
by:

.. ipython:: python

    # compute P(k) from P(k,mu)
    power_2d.average('mu_cen')
    




