API Reference
=============

The full list of nbodykit modules is available :doc:`here <modules>`. We
summarize the most important aspects of the API below.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. _api-io:

The IO Library (:mod:`nbodykit.io`)
-----------------------------------

Base class:

.. autosummary::

  ~nbodykit.io.base.FileType

Subclasses available from the :mod:`nbodykit.io` module:

.. currentmodule:: nbodykit.io

.. autosummary::

  ~bigfile.BigFile
  ~binary.BinaryFile
  ~csv.CSVFile
  ~fits.FITSFile
  ~hdf.HDFFile
  ~stack.FileStack
  ~tpm.TPMBinaryFile


.. _api-cosmology:

Cosmology (:mod:`nbodykit.cosmology`)
-------------------------------------

.. currentmodule:: nbodykit.cosmology

The main cosmology object is

.. autosummary::

  ~cosmology.Cosmology

with available transfer functions computed using

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

The built-in cosmologies are:

===================================== ============================== ====  ===== =======
Name                                  Source                         H0    Om    Flat
===================================== ============================== ====  ===== =======
:attr:`~nbodykit.cosmology.WMAP5`     Komatsu et al. 2009            70.2  0.277 Yes
:attr:`~nbodykit.cosmology.WMAP7`     Komatsu et al. 2011            70.4  0.272 Yes
:attr:`~nbodykit.cosmology.WMAP9`     Hinshaw et al. 2013            69.3  0.287 Yes
:attr:`~nbodykit.cosmology.Planck13`  Planck Collab 2013, Paper XVI  67.8  0.307 Yes
:attr:`~nbodykit.cosmology.Planck15`  Planck Collab 2015, Paper XIII 67.7  0.307 Yes
===================================== ============================== ====  ===== =======

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
  ~array.ArrayCatalog
  ~halos.HaloCatalog
  ~hod.HODCatalog
  ~lognormal.LogNormalCatalog
  ~uniform.UniformCatalog
  ~uniform.RandomCatalog
  ~fkp.FKPCatalog
  ~species.MultipleSpeciesCatalog


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
    ~nbodykit.algorithms.threeptcf.Multipoles3PCF
    ~nbodykit.algorithms.sim_paircount.SimulationBoxPairCount
    ~nbodykit.algorithms.survey_paircount.SurveyDataPairCount
    ~nbodykit.algorithms.survey_paircount.AngularPairCount

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
    nbodykit.utils.JSONEncoder
    nbodykit.utils.JSONDecoder

Generating Mock Data
^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    nbodykit.mockmaker.gaussian_complex_fields
    nbodykit.mockmaker.gaussian_real_fields
    nbodykit.mockmaker.poisson_sample_to_points
