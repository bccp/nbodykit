0.2.1 (unreleased)
------------------

* base dependencies + extras (halotools, h5py); install all dependencies via pip nbodykit[extras]
* meta-data calculations in FKPCatalog now account for Source selection properly
* support for numpy int/float meta-data in JSON output files
* Cosmology instances no longer return attributes as Quantity instances, assuming a default set of units
* renaming of various classes/module related to the nbodykit.Source syntax
    * no more nbodykit.Source in nbodykit.lab
    * nbodykit.source.particle has been renamed to nbodykit.source.catalog
    * source objects are now catalogs -- there class names have "Catalog" appended to their names
    * added individual catalogs for different file types in nbodykit.io, i.e., CSVCatalog, HDFCatalog, etc
* the ``.apply`` operation is no longer in place for sources; it returns a view with the
list of actions extended
