from .base import FileType
from . import tools
from six import string_types
import numpy
try: import fitsio
except ImportError: fitsio = None

class FITSFile(FileType):
    """
    A file object to handle the reading of FITS data using the
    :mod:`fitsio` package.

    See also: https://github.com/esheldon/fitsio

    Parameters
    ----------
    path : str
        the file path to load
    ext: number or string, optional
        The extension.  Either the numerical extension from zero
        or a string extension name. If not sent, data is read from
        the first HDU that has data.
    """
    def __init__(self, path, ext=None):

        # hide the import exception
        if fitsio is None:
            raise ImportError("please install fitsio: ``conda install -c bccp fitsio``")

        self.path = path

        # try to find the first Table HDU to read if not specified
        with fitsio.FITS(path) as ff:
            if ext is None:
                for i, hdu in enumerate(ff):
                    if hdu.has_data():
                        ext = i
                        break
                if ext is None:
                    raise ValueError("input fits file '%s' has not binary table to read" %path)
            else:
                if isinstance(ext, string_types):
                    if ext not in ff:
                        raise ValueError("FITS file does not contain extension with name '%s'" %ext)
                elif ext >= len(ff):
                    raise ValueError("FITS extension %d is not valid" %ext)

            # make sure we crash if data is wrong or missing
            if not ff[ext].has_data() or ff[ext].get_exttype() == 'IMAGE_HDU':
                raise ValueError("FITS extension %d is not a readable binary table" %ext)

        self.attrs = {}
        self.attrs['ext'] = ext

        # size and dtype
        with fitsio.FITS(path) as ff:
            self.size = ff[ext].get_nrows()
            self.dtype = ff[ext].get_rec_dtype()[0]

    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the file

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
        numpy.array
            structured array holding the requested columns over
            the specified range of rows
        """
        if isinstance(columns, string_types): columns = [columns]

        kws = {'ext':self.attrs['ext'], 'columns':columns, 'rows':range(start, stop, step)}
        return fitsio.read(self.path, **kws)
