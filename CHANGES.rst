Changelog
=========

0.3.16 (UNRELEASED)
------------------

0.3.15 (2020-09-08)
------------------
* :issue:`628`: FileType and FileCatalog clean up.
* :issue:`630`: Fix wrong shape (length of 6 instead of 3) of complex mesh stored in attrs.
* :issue:`640`: update for pmesh 0.1.56
* :issue:`641`: update for pandas 1.1.

0.3.14 (2020-07-08)
-------------------
* :issue:`622`: meshtools sets mu to zero instead of nan when appropriate.
* :issue:`618`: zhist fix
* :issue:`613`: gaussian filter fix (was not applied)
* :issue:`610`: odd multipole and wide angle fix in ConvFFTPower
* :issue:`604`: Allow any glob pattern.

0.3.13 (2019-08-01)
-------------------
* :issue:`000`: Update for new dask (csv)
* :issue:`597`: Update docrep.py
* :issue:`596`: Update documentation for NERSC usage. Python 3.5 is no longer supported on NERSC.

0.3.12 (2019-07-06)
-------------------
* :issue:`592`: Update for new dask (ndim)
* :issue:`587`: Reserve label == 0 for unconnected particles.
* :issue:`586`: Center the particles in fiber collision
* :issue:`582`: documentation infra update.

0.3.11 (2019-04-28)
------------------
* :issue:`575,576`: Fix bug in painting particles to mesh in CatalogMesh
* :issue:`569`: Upgrade DistributedArray
* :issue:`570`: Add initial Query interface to FileCatalog.
* :issue:`571`: Typo in FKP
* :issue:`573`: Support Gadget variant binary format (with 4 byte header)
* :issue:`574`: DistributedArray.bincount fix
* :issue:`577`: Add doc examples for CVS and ArrayCatalog.
* :issue:`579`: compat-fix for mcfit-0.0.16
* :issue:`581`: compat-fix for sympy-1.4

0.3.10 (2019-02-07)
------------------
* :issue:`555,556`: Impove dask interpolation in Catalog.save.
* :issue:`557`: Use dask's gufunc for sky transforms
* :issue:`558`: Note the unit of HaloRadius (proper, not comoving. bite me)
* :issue:`561`: Allow setting the position column in 2PCF.
* :issue:`563`: RedshiftHistogram extrapolates to zero rather than any number.
* :issue:`564`: Fix missing compensations with Multi-species meshes.
* :issue:`565`: add keyward header to HDFCatalog for reading additional meta data
* :issue:`566`: fix a bug sorting on float32, and add a persist method to Catalog.
* :issue:`567`: suppress redundant output in convpower.

0.3.9 (2019-01-07)
------------------
* :issue:`544,549`: Use dask.store to save a catalog.
* :issue:`545`: Fix shotnoise estimation of weighted to_mesh() calls.
* :issue:`546`: Fix IndexError in painting during throttling.
* :issue:`547`: Update docrep
* :issue:`548`: Fix many deprecation warnings
* :issue:`550`: Documentation updates
* :issue:`553`: Fix error in TPCF module when there is only 1 bin.

0.3.8 (2018-12-29)
------------------
* :issue:`543`:  Further performance improvements on catalog slicing.
* :issue:`542`:  The IO module shall make sure buffer is c-contiguous before reshaping
* :issue:`541`:  Allow setting cartesian / sphericial transformation reference frame
* :issue:`540`:  Allow not saving the header in Catalog.save
* :issue:`539`:  Allow non-uniform redshifts in halo property transformations.
* :issue:`538`:  Stop gathering catalog to a single rank in HaloCatalog
* :issue:`537`:  Use numpy.sum for summing of integers.
* :issue:`536`:  Fix boxsize mismatch comparision in pair counters.
* :issue:`535`:  Improve working with a dask cluster.
* :issue:`532`:  Improve speed of slicing of a catalog.
* :issue:`531`:  Additional throttling during painting.
* :issue:`530`:  Use setuptools (need to change conda-build-bccp recipe)
* :issue:`529`:  Add kmax(rmax) to FFTPower, FFTCorr, ConvPower.
* :issue:`528`:  Add dataset= to Catalog.save, deprecate datasets=[]

0.3.7 (2018-10-17)
------------------
* :issue:`519`:  Rework the class hierarchy of Catalogmesh.
* :issue:`526`:  Reduce the paint size for systems with lower mem per core
* :issue:`527`:  Aggregate attrs of header and the main datasets.

0.3.6 (2018-09-26)
------------------
* :issue:`518`:  Rework CurrentMPIComm
* :issue:`521`:  Fix OOM errors with dask >= 0.19.0

0.3.5 (2018-08-23)
------------------
* :issue:`509`:  Fix auto detection of f8 type in Gadget1 file reader
* :issue:`513`:  Ignore divide errors.
* :issue:`516`:  Fix several bugs in three point function
* :issue:`517`:  Improve compatibility with numpy 1.15.x's new indexing convention.

0.3.4 (2018-06-29)
------------------
* :issue:`495`:  Improve scaling of LogNormal catalog
* :issue:`497`:  take method to BinnedStatistic
* :issue:`498`:  add compute method to Catalog interface; CatalogMesh no longer a Catalog
* :issue:`500`:  unique binning in FFTPower and FFTCorr
* :issue:`503`:  redistributing a catalog spatially
* :issue:`504`:  Catalog.copy hangs
* :issue:`505`:  update docrep to 0.2.3
* :issue:`506`:  compatible with dask 0.18.1.

0.3.3 (2018-05-30)
------------------
* :issue:`491`:  update compatibility with pandas 0.23.0 in cgm.
* :issue:`490`:  write more useful weights and pairs in the paircount result.
* :issue:`493,494`:  update for deprecation in pmesh

0.3.2 (2018-05-14)
------------------
* :issue:`475`:  proper normalization of the Landy-Szalay estimator, included R1R2 option and to_xil function
* :issue:`487`:  Linear theory correspondant of nbody simulation. (three fluid model)
* :issue:`486`:  overdecomposition in FOF
* :issue:`483`:  switching to a new type in BinnedStatistics.copy()
* :issue:`482`:  Fix a crash when two datasets passed into corrfunc are of different dtypes.
* :issue:`480`:  BigFileCatalog shall look for header relative to the root of file.
* :issue:`479`:  GatherArray allows root=Ellipsis (for allbather)
* :issue:`476`:  Fix MeshSource.apply if MeshSource.action is overriden
* :issue:`471`:  Decompose of surveydata to the correct bounds.

0.3.1 (2018-04-10)
------------------
* :issue:`468`:  corrfunc and big-endian floating point numbers
* :issue:`470`:  Add hankel tranforms for ell>0 
* :issue:`469`:  Fix a regression painting 'apply'ed meshes.

0.3.0 (2017-12-18)
------------------
* :issue:`439`: added updated pair counter algorithms, SurveyDataPairCount and SimulationBoxPairCount.
* :issue:`439`: added correlation function algorithms, SurveyData2PCF and SimulationBox2PCF
* :issue:`441`: add a DemoHaloCatalog for tutorials that downloads small halo catalogs using Halotools
* :issue:`441`: add hod module with wrapper classes for Halotools models and create HOD catalog by calling the populate() method of a HaloCatalog
* :issue:`445`: add a global cache with fixed size for dask calculations
* :issue:`446`: fixes future warning generated by pandas
* :issue:`447`: adds PCS sampling windows

0.2.9 (2017-11-15)
------------------
* :issue:`442`: bug fix: fixes MemoryError when data is larger than memory in paint(); adds `paint_chunk_size` default option
* :issue:`440`: Selection, Value, Weight specified as "default" columns; default columns are not saved to disk
* :issue:`437`: bug fix: make sure to copy attributes of catalog when copy() is called
* :issue:`436`: FFT-based correlation function algorithm, FFTCorr addded
* :issue:`435`: binder badge added to README and documentation for cookbook recipes
* :issue:`433`: by default, the header file will be found automatically in Bigfile
* :issue:`429,432`: updates to documentation
* :issue:`430`: fix bug in FOF due to stricter numpy casting rules in numpy 1.13.3
* :issue:`428`: fixes bug in painting normalization when using interlacing is used
* :issue:`422`: proper list of attributes/methods added to documentation of Cosmology class
* :issue:`425`: latex requirement removed from ``notebook.mplstyle`` style file
* :issue:`423`: support for Gadget 1 file format

0.2.8 (2017-10-06)
------------------

* :issue:`398`: AngularPairCount algorithm added to compute pair counts for survey data as a function of angular separation
* :issue:`364`: fix load balancing for survey pair counting algorithms
* :issue:`415`: fix sympy pickling issue
* :issue:`409`: fix periodic boundary condition issues with FOF for low number of ranks
* :issue:`420`: fix bug introduced in 0.2.7 causing selection of CatalogSources to sometimes hang
* :issue:`420`: remove dask selection optimizations, which can cause the code to crash in uncontrollable ways
* :issue:`421`: better error messaging when using deprecated __init__ syntax for Cosmology class
* :issue:`406`: add global sort and slice operations to CatalogSource objects

0.2.7 (2017-09-25)
------------------

* :issue:`384`: fix packaging bug causing ``notebook.mplstyle`` to be missing from the conda build
* rename test driver from ``runtests.py`` to ``run-tests.py``
* set_options context manager add to set global configuration variables
* :issue:`392,403`: add optimized slicing via dask when applying a boolean selection index to a CatalogSource
* :issue:`393`: CatalogMesh is implemented as a view of a CatalogSource -- column set/gets operate on the underlying CatalogSource
* ConvolvedFFTPower supports cross-correlations of 2 mesh objects originating from the same data/randoms, allowing users to apply different weighting schemes to the two meshes
* transform.SkyToCartesion deprecated in favor of transform.SkyToCartesian
* :issue:`386`: bug fixes related to behavior of Cosmology.clone

0.2.6 (2017-08-29)
------------------

* :issue:`379`: updated Cosmology class built on classylss, a Python binding of the CLASS Boltzmann code
* :issue:`379`: LinearPower object added with CLASS or Eisenstein-Hu transfer
* :issue:`379`: ZeldovichPower object added to compute Zel'dovich power spectrum
* :issue:`379`:HalofitPower object added to compute nonlinear power
* :issue:`379`: CorrelationFunction object added to FFT power spectra to compute theoretical correlation functions
* :issue:`379`: EHPower and NoWiggleEHPower deprecated in favor of LinearPower object

0.2.5 (2017-08-25)
------------------

* :issue:`359`: CSVFile and CSVCatalog no longer fail to read the last line of data when the file does not end in a newline
* :issue:`361`: add CylindricalGroups algorithm for computing groups of objects using the cylindrical grouping method of arXiv:1611.04165
* :issue:`355`: SimulationBoxPairCount and SurveyDataPairCount classes added to perform pair counting of objects in either simulation boxes or from survey data catalogs (using ``Corrfunc`` code)
* :issue:`370`: large addition of documentation for version 0.2.x; still partially completed
* DataSet has been renamed to BinnedStatistic
* calculation of ``dk`` fixed in ProjectedFFTPower
* paint() supports a Nmesh parameter, for easier re-sampling of mesh objects
* :issue:`368`: addition of ``Value`` column for painting mesh objects; this represents the value of the field painted, i.e., unity to paint density, or velocity to paint momentum (number-weighted velocity)
* addition of style module with matplotlib style sheet to make nice plots in our doc tutorials; this makes the docs reproducible by users
* transform.vstack deprecated in favor of transform.StackColumns
* transform.concatenate deprecated in favor of transform.ConcatenateSources
* when painting catalogs to a mesh, users can specify the position column to use via the ``position`` keyword
* :issue:`142`: MultipleSpeciesCatalog object added to support painting multiple species of particles to the same mesh, i.e, baryons and dark matter particles in hydro simulations
* CatalogMeshSource renamed to CatalogMesh internally
* can now delete a column from a CatalogSource
* can now slice a CatalogSource using a list of column names
* :issue:`373`: fix bug in ConstantArray when length is 1

0.2.4 (2017-06-18)
------------------

* :issue:`339`: transform.StackColumns renamed to ``vstack``
* :issue:`339`: transform.concatenate function added, which takes a list of source objects, and returns a new Source that has the concatenation of all data
* :issue:`345`: fix compatibility with halotools version 0.5
* :issue:`346`: ability to resample a MemoryMesh object
* :issue:`344`: bug fixes related to calculation of growth rate in cosmology module
* :issue:`347`: ArrayCatalog can now be initialized from a dictionary or structured array
* :issue:`348`: add a ProjectedFFTPower algorithm, that computes the FFT Power, but can project over certain axes, i.e., projected axes have their power averaged over
* :issue:`353`: FITSCatalog added to the io module, for reading FITS files
* :issue:`352`: KDDensity to quickly estimate local density in density region.
* :issue:`352`: FOF also identifies Peak position and velocity.

0.2.3 (2017-05-19)
------------------

* use of ``resampler`` keyword in the ``paint`` function for compatibility with pmesh versions >= 0.1.24
* bug fixes and code cleanup

0.2.2 (2017-04-27)
------------------

* package maintenance updates only

0.2.1 (2017-04-26)
------------------

* base dependencies + extras (halotools, h5py); install all dependencies via pip nbodykit[extras]
* meta-data calculations in FKPCatalog now account for Source selection properly
* support for numpy int/float meta-data in JSON output files
* Cosmology instances no longer return attributes as Quantity instances, assuming a default set of units
* renaming of various classes/module related to the nbodykit.Source syntax

  - no more nbodykit.Source in nbodykit.lab
  - nbodykit.source.particle has been renamed to nbodykit.source.catalog
  - source objects are now catalogs -- there class names have "Catalog" appended to their names
  - added individual catalogs for different file types in nbodykit.io, i.e., CSVCatalog, HDFCatalog, etc

* the ``.apply`` operation is no longer in place for sources; it returns a view with the list of actions extended
* galaxy type (central vs satellite) stored as integers in HODCatalog
