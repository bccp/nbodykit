Changelog
=========

0.2.8 (unreleased)
------------------


0.2.7 
-----

* packaging bug fixed that caused ``notebook.mplstyle`` to be missing from the conda build
* re-named the test driver ``runtests.py`` to ``run-tests.py``
* set_options context manager add to set global configuration variables
* Optimized slicing via dask when applying a boolean selection index to a CatalogSource
* CatalogMesh is implemented as a view of a CatalogSource -- column set/gets operate on the underlying CatalogSource
* ConvolvedFFTPower supports cross-correlations of 2 mesh objects originating from the same data/randoms, allowing users to apply different weighting schemes to the two meshes
* transform.SkyToCartesion deprecated in favor of transform.SkyToCartesian

0.2.6
-----

* updated Cosmology class built on classylss, a Python binding of the CLASS Boltzmann code
* LinearPower object added with CLASS or Eisenstein-Hu transfer
* ZeldovichPower object added to compute Zel'dovich power spectrum
* HalofitPower object added to compute nonlinear power
* CorrelationFunction object added to FFT power spectra to compute theoretical correlation functions
* EHPower and NoWiggleEHPower deprecated in favor of LinearPower object

0.2.5
-----
* CSVFile and CSVCatalog no longer fails to read the last line of data when the file does not end in a newline
* CylindricalGroups algorithm added for computing groups of objects using the cylindrical grouping method of arXiv:1611.04165
* SimulationBoxPairCount and SurveyDataPairCount classes added to perform pair counting of objects in either simulation boxes or from survey data catalogs (using ``Corrfunc`` code)
* large addition of documentation for version 0.2.x; still partially completed
* DataSet has been renamed to BinnedStatistic
* calculation of ``dk`` fixed in ProjectedFFTPower
* paint() supports a Nmesh parameter, for easier re-sampling of mesh objects
* addition of ``Value`` column for painting mesh objects; this represents the value of the field painted, i.e., unity to paint density, or velocity to paint momentum (number-weighted velocity)
* addition of style module with matplotlib style sheet to make nice plots in our doc tutorials; this makes the docs reproducible by users
* transform.vstack deprecated in favor of transform.StackColumns
* transform.concatenate deprecated in favor of transform.ConcatenateSources
* when painting catalogs to a mesh, users can specify the position column to use via the ``position`` keyword
* MultipleSpeciesCatalog object added to support painting multiple species of particles to the same mesh, i.e, baryons and dark matter particles in hydro simulations
* CatalogMeshSource renamed to CatalogMesh internally
* can now delete a column from a CatalogSource
* can now slice a CatalogSource using a list of column names
* fix bug in ConstantArray when length is 1

0.2.4
-----

* transform.StackColumns renamed to ``vstack``
* transform.concatenate function added, which takes a list of source objects, and returns a new Source that has the concatenation of all data
* compatibility with halotools version 0.5
* ability to resample a MemoryMesh object
* bug fixes related to calculation of growth rate in cosmology module
* ArrayCatalog can now be initialized from a dictionary or structured array
* add a ProjectedFFTPower algorithm, that computes the FFT Power, but can project over certain axes, i.e., projected axes have their power averaged over
* FITSCatalog added to the io module, for reading FITS files
* KDDensity to quickly estimate local density in density region.
* FOF also identifies Peak position and velocity.

0.2.3
------

* use of ``resampler`` keyword in the ``paint`` function for compatibility with pmesh versions >= 0.1.24
* bug fixes and code cleanup

0.2.2
------

* package maintenance updates only

0.2.1
------

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
