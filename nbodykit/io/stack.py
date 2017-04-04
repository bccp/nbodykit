from ..extern.six import string_types
from .base import FileType
from . import tools

import numpy
import os
import inspect

class FileStack(FileType):
    """
    A file object that offers a continuous view of a stack of 
    subclasses of :class:`FileType` instances
    
    This allows data to be accessed across multiple files from
    a single file object
    """    
    def __init__(self, filetype, path, *args, **kwargs):
        """
        Parameters
        ----------
        filetype : FileType subclass
            the type of file class to initialize
        path : str
            list of file names, or string specifying single file or
            containing a glob-like '*' pattern
        *args : 
            additional arguments to pass to the ``filetype`` instance
            during initialization
        **kwargs : 
            additional keyword arguments passed to the ``filetype`` instance
            during initialization
        """
        # check that filetype is subclass of FileType
        if not inspect.isclass(filetype) or not issubclass(filetype, FileType):
            raise ValueError("the stack of `filetype` objects must be subclasses of `FileType`")

        # save the list of relevant files
        if isinstance(path, list):
            filenames = path
        elif isinstance(path, string_types):
            if '*' in path:
                from glob import glob
                filenames = list(map(os.path.abspath, sorted(glob(path))))
            else:
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                filenames = [os.path.abspath(path)]
        else:
            raise ValueError("'path' should be a string or a list of strings")
        self.files = [filetype(fn, **kwargs) for fn in filenames]
        self.sizes = numpy.array([len(f) for f in self.files], dtype='i8')

        # set dtype and size
        self.dtype = self.files[0].dtype
        self.size  = self.sizes.sum()

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
        if isinstance(columns, string_types): columns = [columns]

        toret = []
        for fnum in tools.get_file_slice(self.sizes, start, stop):

            # the local slice
            sl = tools.global_to_local_slice(self.sizes, start, stop, fnum)

            # read this chunk
            toret.append(self.files[fnum].read(columns, sl[0], sl[1], step=1))

        self.logger.debug("Reading column %s [%d:%d] from file %s" % (columns, sl[0], sl[1], self))
        return numpy.concatenate(toret, axis=0)[::step]
