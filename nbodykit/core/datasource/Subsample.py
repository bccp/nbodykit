from nbodykit.core import DataSource
import numpy

class SubsampleDataSource(DataSource):
    """
    Class to read field data from a HDF5 Subsample data file

    Notes
    -----
    * `h5py` must be installed to use this data source.
    """
    plugin_name = "Subsample"

    def __init__(self, path, dataset="Subsample", BoxSize=None, rsd=None):
        try:
            import h5py
        except:
            name = self.__class__.__name__
            raise ImportError("h5py must be installed to use '%s' reader" %name)

        self.path = path
        self.dataset = dataset
        self.BoxSize = BoxSize
        self.rsd = rsd
        
        BoxSize = numpy.empty(3, dtype='f8')
        if self.comm.rank == 0:
            dataset = h5py.File(self.path, mode='r')[self.dataset]
            BoxSize[:] = dataset.attrs['BoxSize']
            self.logger.info("Boxsize from file is %s" % str(BoxSize))
        BoxSize = self.comm.bcast(BoxSize)

        if self.BoxSize is None:
            self.BoxSize = BoxSize
        else:
            if self.comm.rank == 0:
                self.logger.info("Overriding BoxSize as %s" % str(self.BoxSize))

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "read data from a HDF5 Subsample file"

        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("dataset", type=str, help="the name of the dataset in HDF5 file")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="overide the size of the box; can be a scalar or a three-vector")
        s.add_argument("rsd", type=str, choices="xyz",
            help="direction to do redshift distortion")

    def readall(self):
        import h5py

        dataset = h5py.File(self.path, mode='r')[self.dataset]
        data = dataset[...]

        dtype = numpy.dtype([
                ('Position', ('f4', 3)),
                ('Velocity', ('f4', 3))
                ])
        data2 = numpy.empty(len(data),dtype=dtype)

        data2['Position'] = data['Position'] * self.BoxSize
        data2['Velocity'] = data['Velocity'] * self.BoxSize

        nobj = (len(data2), len(data))

        self.logger.info("total number of objects selected is %d / %d" % nobj)

        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            data2['Position'][:, dir] += data2['Velocity'][:, dir]
            data2['Position'][:, dir] %= self.BoxSize[dir]

        return {key: data2[key].copy() for key in data2.dtype.names}
