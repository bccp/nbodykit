from __future__ import absolute_import
# the future import is important. or in python 2.7 we try to 
# import this module itself. Due to the unfortnate name conflict!

import numpy

from . import FileType
from ..extern.six import string_types

class BigFile(FileType):
    """
    A file object to handle the reading of columns of data from 
    a ``bigfile`` file. ``bigfile`` is the default format of 
    FastPM and MP-Gadget.
    
    https://github.com/rainwoodman/bigfile
    """
    plugin_name = "FileType.BigFile"

    def __init__(self, path, exclude=['header'], header='.', root='./'):
        if not root.endswith('/'): root = root + '/'

        if self.comm.rank == 0:
            self.logger.info("Fetching header from %s" % header)
            self.logger.info("Chroot to %s" % root)

        import bigfile
        # the file path
        self.file = bigfile.BigFileMPI(filename=path, comm=self.comm)
        self.root = root

        columns = self.file[self.root].blocks
        columns = list(set(columns) - set(exclude))

        ds = bigfile.BigData(self.file[self.root], columns)

        # set the data type and size
        self.dtype = ds.dtype
        self.size = ds.size
        
        # XXX: important to hold the header block open
        # since attrs probably doesn't hold a reference (I believe).
        self.header = self.file[header]
        
        # store the attributes
        self.attrs = {}
        attrs = self.header.attrs
        for k in attrs.keys():
            self.attrs[k] = attrs[k]

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "A class to read columns of data stored in the `bigfile` format"

        s.add_argument("path", type=str, 
            help='the name of the file to load')
        s.add_argument("exclude", type=str, nargs="+",
            help='columns to exclude')
        s.add_argument("header", type=str,
            help='block to look for the meta data attributes')
        s.add_argument("root", type=str,
            help='block to look for the meta data attributes')

    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range, 
        as a dictionary

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the binary file (in particles)
        """ 
        import bigfile
        if isinstance(columns, string_types): columns = [columns]

        with self.file[self.root] as f:
            ds = bigfile.BigData(f, columns)
            return ds[start:stop][::step]
