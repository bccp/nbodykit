.. currentmodule:: nbodykit.source.catalog

.. _reading-catalogs:

.. ipython:: python
    :suppress:

    import tempfile, os
    startdir = os.path.abspath('.')
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

Reading Catalogs from Disk
==========================

Supported Data Formats
----------------------

nbodykit provides support for initializing
:class:`~nbodykit.base.catalog.CatalogSource` objects by reading tabular data
stored on disk in a variety of formats:

* :ref:`csv-data`
* :ref:`binary-data`
* :ref:`hdf-data`
* :ref:`bigfile-data`
* :ref:`fits-data`

In this section, we provide short examples illustrating how to read data
stored in each of these formats. If your data format is not currently
supported, please see :ref:`custom-data-format`.

.. _csv-data:

Plaintext Data
^^^^^^^^^^^^^^

Reading data stored as columns in plaintext files is supported via the
:class:`~file.CSVCatalog` class. This class partitions the CSV file into chunks, and
data is only read from the relevant chunks of the file, using
the :func:`pandas.read_csv` function. The class accepts any configuration
keywords that this function does. The partitioning step provides a significant
speed-up when reading from the end of the file, since the entirety of the data
does not need to be read first.

.. note::
  By default, the class reads space-separated data with no comments or header
  lines in the input data file.

As an example, here we generate 5 columns for 100 fake objects and write
to a plaintext file:

.. ipython:: python

    import numpy
    from nbodykit.source.catalog import CSVCatalog

    # generate some fake ASCII data
    data = numpy.random.random(size=(100,5))

    # save to a plaintext file
    numpy.savetxt('csv-example.dat', data, fmt='%.7e')

    # name each of the 5 input columns
    names =['a', 'b', 'c', 'd', 'e']

    # read the data
    f = CSVCatalog('csv-example.dat', names)

    print(f)
    print("columns = ", f.columns) # default Weight,Selection also present
    print("total size = ", f.csize)


.. _binary-data:

Binary Data
^^^^^^^^^^^

The :class:`~file.BinaryCatalog` object reads binary data that is stored
on disk in a column-major format. The class can read any numpy data type
and can handle arbitrary byte offsets between columns. However, the column
data must be stored in successive order in the binary file (column-major).

For example,

.. ipython:: python

    from nbodykit.source.catalog import BinaryCatalog

    # generate some fake data and save to a binary file
    with open('binary-example.dat', 'wb') as ff:
        pos = numpy.random.random(size=(1024, 3)) # fake Position column
        vel = numpy.random.random(size=(1024, 3)) # fake Velocity column
        pos.tofile(ff); vel.tofile(ff); ff.seek(0)

    # read the binary data
    f = BinaryCatalog(ff.name, [('Position', ('f8', 3)), ('Velocity', ('f8', 3))], size=1024)

    print(f)
    print("columns = ", f.columns) # default Weight,Selection also present
    print("total size = ", f.csize)

.. _hdf-data:

HDF Data
^^^^^^^^

The :class:`~file.HDFCatalog` object uses the :mod:`h5py` module to read
HDF files. The class supports reading columns stored in :class:`h5py.Dataset`
objects and in :class:`h5py.Group` objects, assuming that all arrays are of the
same length since the catalog has a fixed size. Columns stored in different
datasets or groups can be accessed via their full path in the HDF file.

In the example below, we load fake data from both the dataset "Data1" and
from the group "Data2" in an example HDF5 file:

.. ipython:: python

    import h5py
    from nbodykit.source.catalog import HDFCatalog

    # generate some fake data
    dset = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Mass', 'f8')])
    dset['Position'] = numpy.random.random(size=(1024, 3))
    dset['Mass'] = numpy.random.random(size=1024)

    # write to a HDF5 file
    with h5py.File('hdf-example.dat' , 'w') as ff:
        ff.create_dataset('Data1', data=dset)
        grp = ff.create_group('Data2')
        grp.create_dataset('Position', data=dset['Position']) # column as dataset
        grp.create_dataset('Mass', data=dset['Mass']) # column as dataset

    # read the data
    f = HDFCatalog('hdf-example.dat')

    print(f)
    print("columns = ", f.columns) # default Weight,Selection also present
    print("total size = ", f.csize)

.. _bigfile-data:

Bigfile Data
^^^^^^^^^^^^

The `bigfile <https://github.com/rainwoodman/bigfile>`_ package is a massively
parallel IO library for large, hierarchical datasets, and nbodykit supports
reading data stored in this format using :class:`~file.BigFileCatalog`.

Below, we read "Position" and "Velocity" columns, stored in the :mod:`bigfile`
format:

.. ipython:: python

    import bigfile
    from nbodykit.source.catalog import BigFileCatalog

    # generate some fake data
    data = numpy.empty(512, dtype=[('Position', ('f8', 3)), ('Velocity', ('f8',3))])
    data['Position'] = numpy.random.random(size=(512, 3))
    data['Velocity'] = numpy.random.random(size=(512,3))

    # save fake data to a BigFile
    with bigfile.BigFile('bigfile-example', create=True) as tmpff:
        with tmpff.create("Position", dtype=('f4', 3), size=512) as bb:
            bb.write(0, data['Position'])
        with tmpff.create("Velocity", dtype=('f4', 3), size=512) as bb:
            bb.write(0, data['Velocity'])
        with tmpff.create("Header") as bb:
            bb.attrs['Size'] = 512.

    # read the data
    f = BigFileCatalog('bigfile-example', header='Header')

    print(f)
    print("columns = ", f.columns) # default Weight,Selection also present
    print("total size = ", f.csize)

.. _fits-data:

FITS Data
^^^^^^^^^

The `FITS <https://fits.gsfc.nasa.gov>`_ data format is supported via the
:class:`~file.FITSCatalog` object. nbodykit relies on the
`fitsio <https://github.com/esheldon/fitsio>`_ package to perform the read
operation.

For example, below we read "Position" and "Velocity" data from a FITS file:

.. ipython:: python

    import fitsio
    from nbodykit.source.catalog import FITSCatalog

    # generate some fake data
    dset = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Mass', 'f8')])
    dset['Position'] = numpy.random.random(size=(1024, 3))
    dset['Mass'] = numpy.random.random(size=1024)

    # write to a FITS file using fitsio
    fitsio.write('fits-example.dat', dset, extname='Data')

    # read the data
    f = FITSCatalog('fits-example.dat', ext='Data')

    print(f)
    print("columns = ", f.columns) # default Weight,Selection also present
    print("total size = ", f.csize)

.. _reading-multiple-files:

Reading Multiple Data Files at Once
-----------------------------------

:class:`~nbodykit.base.catalog.CatalogSource` objects support reading
multiple files at once, providing a continuous view of each individual catalog
stacked together. Each file read must contain the same data types, otherwise
the data cannot be combined into a single catalog.

This becomes particularly useful when the user has data
split into multiple files in a single directory, as is often the case when
processing large amounts of data. For example, output binary snapshots from
N-body simulations, often totaling 10GB - 100GB in size, can be read into a
single :class:`~file.BinaryCatalog` with nbodykit.

When specifying multiple files to load, the user can use either an explicit
list of file names or use an asterisk glob pattern to match files.
As an example, below, we read data from two plaintext files into a single
:class:`~file.CSVCatalog`:

.. ipython:: python

    # generate data
    data = numpy.random.random(size=(100,5))

    # save the first plaintext data file
    numpy.savetxt('csv-example-1.dat', data, fmt='%.7e')

    # and the second plaintext data file
    numpy.savetxt('csv-example-2.dat', data, fmt='%.7e')


Using a glob pattern
^^^^^^^^^^^^^^^^^^^^

.. ipython:: python

    # the names of the columns in both files
    names =['a', 'b', 'c', 'd', 'e']

    # read with a glob pattern
    f = CSVCatalog('csv-example-*', names)

    print(f)

    # combined catalog size is 100+100=200
    print("total size = ", f.csize)

Using a list of file names
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ipython:: python

    # the names of the columns in both files
    names =['a', 'b', 'c', 'd', 'e']

    # read with a list of the file names
    f = CSVCatalog(['csv-example-1.dat', 'csv-example-2.dat'], names)

    print(f)

    # combined catalog size is 100+100=200
    print("total size = ", f.csize)

.. _custom-data-format:

Reading a Custom Data Format
----------------------------

.. currentmodule:: nbodykit.io

Users can implement their own subclasses of :class:`CatalogSource` for reading
custom data formats with a few easy steps. The core functionality of the
:class:`CatalogSource` classes described in this section use the
:mod:`nbodykit.io` module for reading data from disk. This module implements the
:class:`nbodykit.io.base.FileType` base class, which is an abstract
class that behaves like a :obj:`file`-like object. For the built-in
file formats discussed in this section, we have implemented the following
subclasses of :class:`~nbodykit.io.base.FileType` in the :mod:`nbodykit.io`
module: :class:`~csv.CSVFile`, :class:`~binary.BinaryFile`,
:class:`~bigfile.BigFile`, :class:`~hdf.HDFFile`, and :class:`~fits.FITSFile`.

To make a valid subclass of :class:`~nbodykit.io.base.FileType`, users must:

#. Implement the :func:`~nbodykit.io.base.FileType.read` function that reads
   a range of the data from disk.
#. Set the :attr:`size` in the :func:`__init__` function, specifying the total
   size of the data on disk.
#. Set the :attr:`dtype` in the :func:`__init__` function, specifying the type
   of data stored on disk.

Once we have the custom subclass implemented, the
:func:`nbodykit.source.catalog.file.FileCatalogFactory` function can
be used to automatically create a custom :class:`CatalogSource` object
from the subclass.

As a toy example, we will illustrate how this is done for data saved
using the numpy ``.npy`` format. First, we will implement our
subclass of the :class:`~nbodykit.io.base.FileType` class:

.. ipython:: python

    from nbodykit.io.base import FileType

    class NPYFile(FileType):
        """
        A file-like object to read numpy ``.npy`` files
        """
        def __init__(self, path):
            self.path = path
            self.attrs = {}
            # load the data and set size and dtype
            self._data = numpy.load(self.path)
            self.size = len(self._data) # total size
            self.dtype = self._data.dtype # data dtype
        def read(self, columns, start, stop, step=1):
            """
            Read the specified column(s) over the given range
            """
            return self._data[start:stop:step]

And now generate the subclass of :class:`SourceCatalog`:

.. ipython:: python

    from nbodykit.source.catalog.file import FileCatalogFactory

    NPYCatalog = FileCatalogFactory('NPYCatalog', NPYFile)

And finally, we will generate some fake data, save it to a ``.npy`` file,
and then load it with our new ``NPYCatalog`` class:

.. ipython:: python

    # generate the fake data
    data = numpy.empty(1024, dtype=[('Position', ('f8', 3)), ('Mass', 'f8')])
    data['Position'] = numpy.random.random(size=(1024, 3))
    data['Mass'] = numpy.random.random(size=1024)

    # save to a npy file
    numpy.save("npy-example.npy", data)

    # and now load the data
    f = NPYCatalog("npy-example.npy")

    print(f)
    print("columns = ", f.columns) # default Weight,Selection also present
    print("total size = ", f.csize)

This toy example illustrates how custom data formats can be incorporated
into nbodykit, but users should take care to optimize their storage
solutions for more complex applications. In particulars, data storage formats
that are stored in column-major format and allow data slices from arbitrary
locations to be read should be favored. This enables large speed-ups when
reading data in parallel. On the contrary, our simple toy example class
:class:`NPYFile` reads the entirety of the data before returning
a certain slice in the :func:`read` function. In general, this should be
avoided if at all possible.


.. ipython:: python
    :suppress:

    import shutil
    os.chdir(startdir)
    shutil.rmtree(tmpdir)
