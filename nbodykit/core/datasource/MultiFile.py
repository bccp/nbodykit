from nbodykit.core import DataSource
from nbodykit.io.stack import FileStack
import numpy

class MultiFileDataSource(DataSource):
    plugin_name = "MultiFile"

    def __init__(self, filetype, path, args={}, transform={}):
        from nbodykit import plugin_manager
        filetype = plugin_manager.get_plugin(filetype)
        self.cat = FileStack(path, filetype, **args)
        self.transform = transform

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "read snapshot files a multitype file"
        s.add_argument("filetype", help="the file path to load the data from")
        s.add_argument("path", help="the file path to load the data from")
        s.add_argument("args", type=dict, help="the file path to load the data from")
        s.add_argument("transform", type=dict, help="transformation")

    def parallel_read(self, columns, full=False):
        start = self.comm.rank * self.cat.size // self.comm.size
        end = (self.comm.rank  + 1) * self.cat.size // self.comm.size
        # no dask yet!
        ds = self.cat[start:end]
        def ev(column):
            if column in self.transform:
                g = {'ds' : ds}
                return eval(self.transform[column], g)
            else:
                return ds[column]

        yield [ev(key) for key in columns]
