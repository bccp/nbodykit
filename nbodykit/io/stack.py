from .base import FileType
from . import tools
from six import string_types
import numpy
import os
from glob import glob
import inspect

class FileStack(FileType):
    """
    A file object that offers a continuous view of a stack of subclasses of
    :class:`~nbodykit.io.base.FileType` instances.

    This allows data to be accessed across multiple files from
    a single file object. The "stack" is a concatenation
    of one file to the end of the previous file.

    Parameters
    ----------
    filetype : subclass of :class:`~nbodykit.io.base.FileType`
        the type of file class to initialize
    path : str
        list of file names, or string specifying single file or
        a glob pattern.
    *args :
        additional arguments to pass to the ``filetype`` instance
        during initialization
    **kwargs :
        additional keyword arguments passed to the ``filetype`` instance
        during initialization
    """
    def __init__(self, filetype, path, *args, **kwargs):

        # check that filetype is subclass of FileType
        if not inspect.isclass(filetype) or not issubclass(filetype, FileType):
            raise ValueError("the stack of `filetype` objects must be subclasses of `FileType`")

        self.path = path

        # save the list of relevant files
        if isinstance(path, list):
            filenames = path
        elif isinstance(path, string_types):
            filenames = list(map(os.path.abspath, sorted(glob(path))))
            if len(filenames) == 0:
                raise FileNotFoundError(path)
        else:
            raise ValueError("'path' should be a string or a list of strings")
        self.files = [filetype(fn, *args, **kwargs) for fn in filenames]
        self.sizes = numpy.array([len(f) for f in self.files], dtype='i8')

        # set dtype and size
        FileType.__init__(self, dtype=self.files[0].dtype,
                                size=self.sizes.sum())

    def __repr__(self):
        return "FileStack(%s, ... %d files)" % (repr(self.files[0]), self.nfiles)

    @property
    def attrs(self):
        """
        Dictionary of meta-data for the stack
        """
        if hasattr(self.files[0], 'attrs'):
            return self.files[0].attrs
        else:
            return {}

    @property
    def nfiles(self):
        """
        The number of files in the FileStack
        """
        return len(self.files)

    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range,
        returning a structured numpy array

        Parameters
        ----------
        columns : str, list of str
            the name of the column(s) to return
        start : int
            the row integer to start reading at
        stop : int
            the row integer to stop reading at
        step : int, optional
            the step size to use when reading; default is 1

        Returns
        -------
        data : array_like
            a numpy structured array holding the requested data
        """

        toret = []
        for fnum in tools.get_file_slice(self.sizes, start, stop):

            # the local slice
            sl = tools.global_to_local_slice(self.sizes, start, stop, fnum)

            # read this chunk
            toret.append(self.files[fnum].read(columns, sl[0], sl[1], step=1))

        self.logger.debug("Reading column %s [%d:%d] from file %s" % (columns, sl[0], sl[1], self))
        return numpy.concatenate(toret, axis=0)[::step]
