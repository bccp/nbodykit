DataSet for Algorithm results
=============================

Several nbodykit algorithms compute two-point clustering statistics, 
and we provide the :class:`~nbodykit.dataset.DataSet` class for analyzing these results. 
The class is designed to hold data variables at fixed coordinates, 
i.e., a grid of (r, mu) or (k, mu) bins.
    
The DataSet class is modeled after the syntax of :class:`xarray.Dataset`, and we
provide subclasses of DataSet that are specifically designed to hold correlation 
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

Loading results
---------------

Let's create a simple example dataset:

.. ipython:: python
   :suppress:
   
   from nbodykit.test import download_results_file
   download_results_file('test_power_plaintext.dat', '../examples/output') # 2d result
   download_results_file('test_power_cross.dat', '../examples/output') # 1d result
   cd ..
    
.. ipython:: python

    from nbodykit import dataset, files
    
    # output file of 'examples/power/test_plaintext.params'
    filename_2d = 'examples/output/test_power_plaintext.dat'
    
    # load a 2D power result
    power_2d =  dataset.Power2dDataSet.from_nbkit(*files.Read2DPlainText(filename_2d))
    
    # output file of 'examples/power/test_cross_power.params'
    filename_1d = 'examples/output/test_power_cross.dat'
    
    # load a 1D power result
    power_1d =  dataset.Power1dDataSet.from_nbkit(*files.Read1DPlainText(filename_1d))
    
We can select individual mu columns vias

.. ipython:: python

    power_2d[:,0] # first mu column


