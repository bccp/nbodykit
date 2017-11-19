API Reference
=============

The full list of nbodykit modules is available :doc:`here <modules>`. We
summarize the most important aspects of the API below.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. _api-lab:

The nbodykit lab
----------------

To make things easier for users, we import all of the
classes and modules needed to do cool science into a single module:

.. autosummary::

    nbodykit.lab


.. _api-cosmology:

Cosmology (:mod:`nbodykit.cosmology`)
-------------------------------------

.. currentmodule:: nbodykit.cosmology

The main cosmology object relies on the functionality of the :mod:`classylss`
package, which provides a binding of the `CLASS CMB Boltzmann code <http://class-code.net>`_.
The syntax largely follows that used by CLASS. Below, we list the main cosmology class,
as well as its attributes and methods:

.. autosummary::

    ~cosmology.Cosmology

.. rubric:: Attributes

.. autocosmosummary:: nbodykit.cosmology.cosmology.Cosmology
    :attributes:

.. rubric:: Methods

.. autocosmosummary:: nbodykit.cosmology.cosmology.Cosmology
    :methods:


There are several transfer functions available to the user:

.. autosummary::

  ~power.transfers.CLASS
  ~power.transfers.EisensteinHu
  ~power.transfers.NoWiggleEisensteinHu

There are several power spectrum classes

.. autosummary::

  ~power.linear.LinearPower
  ~power.halofit.HalofitPower
  ~power.zeldovich.ZeldovichPower

And a correlation function class and functions for transforming between
power spectra and correlation functions

.. autosummary::

  ~correlation.CorrelationFunction
  ~correlation.xi_to_pk
  ~correlation.pk_to_xi


We also have a class for computing LPT background calculations:

.. autosummary::

  ~background.PerturbationGrowth

.. _builtin-cosmos:

The built-in cosmologies are:

.. include:: builtin-cosmos.rst

.. _api-transform:

Transforming Catalog Data (:mod:`nbodykit.transform`)
------------------------------------------------------

.. autosummary::

  ~nbodykit.transform.ConcatenateSources
  ~nbodykit.transform.StackColumns
  ~nbodykit.transform.ConstantArray
  ~nbodykit.transform.SkyToUnitSphere
  ~nbodykit.transform.SkyToCartesian
  ~nbodykit.transform.HaloConcentration
  ~nbodykit.transform.HaloRadius


Data Sources
------------

.. _api-discrete-data:

Discrete Objects
^^^^^^^^^^^^^^^^

Base class:

.. autosummary::

  ~nbodykit.base.catalog.CatalogSource

And subclasses:

.. currentmodule:: nbodykit.source.catalog

.. autosummary::

  ~file.CSVCatalog
  ~file.BinaryCatalog
  ~file.BigFileCatalog
  ~file.TPMBinaryCatalog
  ~file.HDFCatalog
  ~file.FITSCatalog
  ~file.Gadget1Catalog
  ~array.ArrayCatalog
  ~halos.HaloCatalog
  ~lognormal.LogNormalCatalog
  ~uniform.UniformCatalog
  ~uniform.RandomCatalog
  ~fkp.FKPCatalog
  ~species.MultipleSpeciesCatalog
  ~nbodykit.tutorials.DemoHaloCatalog

Interpolating Objects to a Mesh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The base class:

.. autosummary::

    ~nbodykit.base.catalogmesh.CatalogMesh

And subclasses:

.. autosummary::

    ~nbodykit.source.catalogmesh.species.MultipleSpeciesCatalogMesh
    ~nbodykit.source.catalogmesh.fkp.FKPCatalogMesh


.. _api-mesh-data:

Data Directly on a Mesh
^^^^^^^^^^^^^^^^^^^^^^^

Base class:

.. autosummary::

  ~nbodykit.base.mesh.MeshSource

And subclasses:

.. autosummary::

  ~nbodykit.source.mesh.bigfile.BigFileMesh
  ~nbodykit.source.mesh.linear.LinearMesh
  ~nbodykit.source.mesh.field.FieldMesh
  ~nbodykit.source.mesh.array.ArrayMesh

.. _api-algorithms:

Algorithms (:mod:`nbodykit.algorithms`)
---------------------------------------

.. _api-clustering-statistics:

Clustering Statistics
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~nbodykit.algorithms.fftpower.FFTPower
    ~nbodykit.algorithms.fftpower.ProjectedFFTPower
    ~nbodykit.algorithms.convpower.ConvolvedFFTPower
    ~nbodykit.algorithms.fftcorr.FFTCorr
    ~nbodykit.algorithms.pair_counters.simbox.SimulationBoxPairCount
    ~nbodykit.algorithms.pair_counters.mocksurvey.SurveyDataPairCount
    ~nbodykit.algorithms.paircount_tpcf.tpcf.SimulationBox2PCF
    ~nbodykit.algorithms.paircount_tpcf.tpcf.SurveyData2PCF
    ~nbodykit.algorithms.threeptcf.SimulationBox3PCF
    ~nbodykit.algorithms.threeptcf.SurveyData3PCF

Grouping Methods
^^^^^^^^^^^^^^^^

.. autosummary::

    ~nbodykit.algorithms.fof.FOF
    ~nbodykit.algorithms.cgm.CylindricalGroups
    ~nbodykit.algorithms.fibercollisions.FiberCollisions

Miscellaneous
^^^^^^^^^^^^^

.. autosummary::

    ~nbodykit.algorithms.kdtree.KDDensity
    ~nbodykit.algorithms.zhist.RedshiftHistogram

Managing Multiple Tasks (:class:`~nbodykit.batch.TaskManager`)
--------------------------------------------------------------

.. currentmodule:: nbodykit.batch

.. autosummary::

    TaskManager
    TaskManager.iterate
    TaskManager.map

Analyzing Results (:class:`~nbodykit.binned_statistic.BinnedStatistic`)
-----------------------------------------------------------------------

.. currentmodule:: nbodykit.binned_statistic

.. autosummary::

    BinnedStatistic
    BinnedStatistic.from_json
    BinnedStatistic.to_json
    BinnedStatistic.copy
    BinnedStatistic.rename_variable
    BinnedStatistic.average
    BinnedStatistic.reindex
    BinnedStatistic.sel
    BinnedStatistic.squeeze


.. _api-io:

The IO Library (:mod:`nbodykit.io`)
-----------------------------------

Base class:

.. autosummary::

  ~nbodykit.io.base.FileType

Subclasses available from the :mod:`nbodykit.io` module:


.. autosummary::

  ~nbodykit.io.bigfile.BigFile
  ~nbodykit.io.binary.BinaryFile
  ~nbodykit.io.csv.CSVFile
  ~nbodykit.io.fits.FITSFile
  ~nbodykit.io.hdf.HDFFile
  ~nbodykit.io.stack.FileStack
  ~nbodykit.io.tpm.TPMBinaryFile
  ~nbodykit.io.gadget.Gadget1File

Internal Nuts and Bolts
------------------------

MPI Utilities
^^^^^^^^^^^^^

.. autosummary::

    nbodykit.CurrentMPIComm
    nbodykit.CurrentMPIComm.enable
    nbodykit.CurrentMPIComm.get
    nbodykit.CurrentMPIComm.set
    nbodykit.utils.GatherArray
    nbodykit.utils.ScatterArray

General Utilities
^^^^^^^^^^^^^^^^^

.. autosummary::

    nbodykit.setup_logging
    nbodykit.set_options
    nbodykit.utils.JSONEncoder
    nbodykit.utils.JSONDecoder

Generating Mock Data
^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    nbodykit.mockmaker.gaussian_complex_fields
    nbodykit.mockmaker.gaussian_real_fields
    nbodykit.mockmaker.poisson_sample_to_points
