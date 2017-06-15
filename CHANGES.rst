0.2.4 (unreleased)
------------------

* transform.StackColumns renamed to ``vstack``
* transform.concatenate function added, which takes a list of source objects,
and returns a new Source that has the concatenation of all data
* compatibility with halotools version 0.5
* ability to resample a MemoryMesh object
* bug fixes related to calculation of growth rate in cosmology module
* ArrayCatalog can now be initialized from a dictionary or structured array
* add a ProjectedFFTPower algorithm, that computes the FFT Power, but can
project over certain axes, i.e., projected axes have their power averaged over
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
