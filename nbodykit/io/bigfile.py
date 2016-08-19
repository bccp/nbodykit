from __future__ import absolute_import
# the future import is important. or in python 2.7 we try to 
# import this module itself. Due to the unfortnate name conflict!

import numpy

from . import FileType
from ..extern.six import string_types

class BigFile(FileType):
    """
    A file object to handle the reading of columns of data from 
    a bigfile file. bigfile is the default format of FastPM and MP-Gadget.

       https://github.com/rainwoodman/bigfile

    """
    plugin_name = "FileType.BigFile"

    def __init__(self, path, exclude=['header']):
        import bigfile
        # the file path
        self.file = bigfile.BigFileMPI(filename=path, comm=self.comm)
        columns = self.file.blocks
        columns = list(set(columns) - set(exclude))

        ds = bigfile.BigData(self.file, columns)

        # set the data type
        self.dtype = ds.dtype
        self.size = ds.size

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "a binary file reader"

        s.add_argument("path", type=str, 
            help='the name of the binary file to load')
        s.add_argument("exclude", type=str, nargs="+",
            help='columns to exclude')

    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range, 
        as a dictionary

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the binary file (in particles)
        """ 
        import bigfile
        if isinstance(columns, string_types): columns = [columns]

        ds = bigfile.BigData(self.file, columns)

        return ds[start:stop][::step]
