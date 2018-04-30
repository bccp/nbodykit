from __future__ import absolute_import
# the future import is important. or in python 2.7 we try to
# import this module itself. Due to the unfortnate name conflict!

import numpy

from .base import FileType
from six import string_types
import json
from nbodykit.utils import JSONDecoder

class Automatic: pass

class BigFile(FileType):
    """
    A file object to handle the reading of columns of data from
    a :mod:`bigfile` file.

    :mod:`bigfile` is a reproducible, massively parallel IO library for
    large, hierarchical datasets, and it is the default format of the
    `FastPM <https://github.com/rainwoodman/fastpm>`_ and the
    `MP-Gadget <https://github.com/bluetides-project/MP-Gadget>`_
    simulations.

    See also: https://github.com/rainwoodman/bigfile

    Parameters
    ----------
    path : str
        the name of the directory holding the bigfile data
    exclude : list of str, optional
        the data sets to exlude from loading within bigfile; default
        is the header. If any list is given, the name of the header column
        must be given too if it is not part of the data set.
    header : str, optional
        the path to the header; default is to use a column 'Header'.
        It is relative to the file, not the dataset.
    dataset : str
        finding columns from a specific dataset in the bigfile;
        the default is start looking for columns from the root.
    """
    def __init__(self, path, exclude=None, header=Automatic, dataset='./'):

        if not dataset.endswith('/'): dataset = dataset + '/'

        import bigfile

        self.dataset = dataset
        self.path = path

        # store the attributes
        self.attrs = {}

        # the file path
        with bigfile.BigFile(filename=path) as ff:
            columns = ff[self.dataset].blocks
            header = self._find_header(header, ff)

            if exclude is None:
                # by default exclude header only.
                exclude = [header]

            columns = list(set(columns) - set(exclude))

            ds = bigfile.BigData(ff[self.dataset], columns)

            # set the data type and size
            self.dtype = ds.dtype
            self.size = ds.size

            header = ff[header]
            attrs = header.attrs

            # copy over the attrs
            for k in attrs.keys():

                # load a JSON representation if str starts with json:://
                if isinstance(attrs[k], string_types) and attrs[k].startswith('json://'):
                    self.attrs[k] = json.loads(attrs[k][7:], cls=JSONDecoder)
                # copy over an array
                else:
                    self.attrs[k] = numpy.array(attrs[k], copy=True)

    def _find_header(self, header, ff):
        """ Find header from the file block by default. """
        if header is Automatic:
            for header in ['Header', 'header', '.']:
                if header in ff.blocks: break

        # shall not make the assertion here because header can be nested deep.
        # then not shown in ff.blocks. try catch may work better.
        #if not header in ff.blocks:
        #    raise KeyError("header block `%s` is not defined in the bigfile. Candidates can be `%s`"
        #            % (header, str(ff.blocks))

        return header

    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range,
        as a dictionary

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the binary file (in particles)
        """
        import bigfile
        if isinstance(columns, string_types): columns = [columns]

        with bigfile.BigFile(filename=self.path)[self.dataset] as f:
            ds = bigfile.BigData(f, columns)
            return ds[start:stop][::step]
