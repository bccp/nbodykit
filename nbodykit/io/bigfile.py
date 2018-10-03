from __future__ import absolute_import
# the future import is important. or in python 2.7 we try to
# import this module itself. Due to the unfortnate name conflict!

import numpy

from .base import FileType
from six import string_types
import json
from nbodykit.utils import JSONDecoder
from fnmatch import fnmatch


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
        must be given too if it is not part of the data set. The names
        are shell glob patterns.

    header : str, or list, optional
        the path to the header; default is to use a column 'Header'.
        It is relative to the file, not the dataset.
        If a list is provided, the attributes is updated from the first entry to the last.

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
            columns = [block for block in ff[self.dataset].blocks]
            headers = self._find_headers(header, dataset, ff)

            if exclude is None:
                # by default exclude header only.
                exclude = headers

            if not isinstance(exclude, (list, tuple)):
                exclude = [exclude]

            columns = [
                column
                for column in set(columns) if not any(fnmatch(column, e) for e in exclude)
                ]

            ds = bigfile.BigData(ff[self.dataset], columns)

            # set the data type and size
            self.dtype = ds.dtype
            self.size = ds.size

            headers = [ff[header] for header in headers]
            all_attrs = [ header.attrs for header in headers ]
            for attrs in all_attrs:
                # copy over the attrs
                for k in attrs.keys():

                    # load a JSON representation if str starts with json:://
                    if isinstance(attrs[k], string_types) and attrs[k].startswith('json://'):
                        self.attrs[k] = json.loads(attrs[k][7:], cls=JSONDecoder)
                    # copy over an array
                    else:
                        self.attrs[k] = numpy.array(attrs[k], copy=True)

    def _find_headers(self, header, dataset, ff):
        """ Find header from the file block by default. """
        if header is Automatic:
            header = ['Header', 'header', '.']

        if not isinstance(header, (tuple, list)):
            header = [header]

        r = []
        for h in header:
            if h in ff.blocks:
                if h not in r:
                    r.append(h)

        # append the dataset itself
        r.append(dataset.strip('/') + '/.')

        # shall not make the assertion here because header can be nested deep.
        # then not shown in ff.blocks. try catch may work better.
        #if not header in ff.blocks:
        #    raise KeyError("header block `%s` is not defined in the bigfile. Candidates can be `%s`"
        #            % (header, str(ff.blocks))

        return r

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
