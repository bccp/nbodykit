Changelog
=========

0.2.10 (unreleased)
-------------------

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
