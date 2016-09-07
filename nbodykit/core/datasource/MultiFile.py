from nbodykit.core import DataSource
from nbodykit.io.stack import FileStack
import numpy
class MultiFileDataSource(DataSource):
    plugin_name = "MultiFile"

    def __init__(self, filetype, path, args={}, transform={}, enable_dask=False):
        # cannot do this in the module because the module file is ran before plugin_manager
        # is init.

        from nbodykit import plugin_manager
        filetype = plugin_manager.get_plugin(filetype)
        self.logger.info("Extra arguments to FileType: %s " % args)
        self.cat = FileStack(path, filetype, **args)
        self.transform = transform
        self.enable_dask = enable_dask
        self.attrs = self.cat.attrs

        self.logger.info("attrs = %s" % self.attrs)
        if enable_dask:
            import dask

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "read snapshot files a multitype file"
        s.add_argument("filetype", help="the file path to load the data from")
        s.add_argument("path", help="the file path to load the data from")
        s.add_argument("args", type=dict, help="the file path to load the data from")
        s.add_argument("transform", type=dict, help="transformation")
        s.add_argument("enable_dask", type=bool, help="use dask")

    def parallel_read(self, columns, full=False):
        start = self.comm.rank * self.cat.size // self.comm.size
        end = (self.comm.rank  + 1) * self.cat.size // self.comm.size
        if self.enable_dask:
            import dask.array as da
            ds = da.from_array(self.cat, 1024 * 32)

            def ev(column):
                if column in self.transform:
                    g = {'ds' : ds}
                    return eval(self.transform[column], g)[start:end]
                else:
                    return ds[column][start:end]

            yield [ev(key).compute() for key in columns]
        else:
            ds = self.cat[start:end]
            def ev(column):
                if column in self.transform:
                    g = {'ds' : ds}
                    return eval(self.transform[column], g)
                else:
                    return ds[column]

            yield [ev(key) for key in columns]
