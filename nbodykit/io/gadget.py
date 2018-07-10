from .binary import BinaryFile
import numpy
from six import string_types
from . import tools
import warnings

DefaultColumnDefs = [
    ('Position', ('auto', 3), 'all', ),
    ('GadgetVelocity',  ('auto', 3), 'all', ),
    ('ID', 'auto', 'all', ),
    ('Mass', 'auto', None, ),
    ('InternalEnergy', 'auto', (0, ), ),
    ('Density', 'auto', (0, ), ),
    ('SmoothingLength', 'auto', (0,) ),
    ]

DefaultHeaderDtype = [
        ('Npart', ('u4', 6)),
        ('Massarr', ('f8', 6)),
        ('Time', ('f8')),
        ('Redshift', ('f8')),
        ('FlagSfr', ('i4')),
        ('FlagFeedback', ('i4')),
        ('Nall', ('u4', 6)),
        ('FlagCooling', ('i4')),
        ('NumFiles', ('i4')),
        ('BoxSize', ('f8')),
        ('Omega0', ('f8')),
        ('OmegaLambda', ('f8')),
        ('HubbleParam', ('f8')),
        ('FlagAge', ('i4')),
        ('FlagMetals', ('i4')),
        ('NallHW', ('u4', 6)),
        ('flag_entr_ics', ('i4')),
    ]

class Gadget1File(BinaryFile):
    """
    Read snapshot binary files from Volkers Gadget 1/2/3 simulations.

    These files are stored column-wise with a format, with a
    header of size 28 bytes to begin the file.

    The columns are:

    * Position : 'f4', 'f8' precision
        the position data, usually in Kpc/h units.
    * GadgetVelocity : 'f4', 'f8' precision
        the Gadget 1 velocity, sqrt(a)**-1 v_p. in km/s
    * ID : 'i8'/'i4' precision
        integers specfiying the particle ID

    Parameters
    ----------
    path : str
        the path to the binary file to load
    columndefs : list
        a list of triplets (columnname, element_dtype, particle_types)
    ptype : int
        type of particle of interest.
    hdtype : list, dtype
        dtype of the header; must define Massarr and Npart

    References
    ----------
    https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf
    """
    def __init__(self, path, columndefs=DefaultColumnDefs,
                hdtype=DefaultHeaderDtype, ptype=1):

        if ptype not in [0, 1, 2, 3, 4, 5]:
            raise ValueError("ptype shall be 0 ~ 5.")

        hdtype = numpy.dtype(hdtype)
        hdtype_padded = numpy.dtype([
                                     ('f77', 'i4'),
                                     ('header', hdtype),
                                     ('padding', ('u1', 256 - hdtype.itemsize))])
        header = numpy.fromfile(path, dtype=hdtype_padded, count=1)[0]['header']

        attrs = {}

        for key in header.dtype.names:
            attrs[key] = header[key].copy()

        self.attrs = attrs

        self.file_header = header
        self.header_mass = header['Massarr'][ptype]
        self.ptype = ptype

        dtype = []
        defs = []

        with open(path, 'r') as ff:
            offsets = {}
            ptr = 256 + 4 + 4
            for column, spec, ptypes in columndefs:
                if not isinstance(spec, tuple):
                    spec = spec, ()
                if len(spec) == 1:
                    spec = spec[0], ()

                if ptypes == 'all':
                    ptypes = [0, 1, 2, 3, 4, 5]
                elif column == 'Mass':
                    ptypes = (header['Massarr'] == 0).nonzero()[0]

                blocksize = 0
                reloffset = 0
                N = 0
                for i in ptypes:
                    if i == ptype:
                        reloffset = N
                    N += int(header['Npart'][i])

                if N != 0: # block exists
                    ff.seek(ptr, 0)
                    a = numpy.fromfile(ff, dtype='i4', count=1)[0]
                    ptr += 4

                    itemsize = a // N # compute precision from blocksize

                    blocksize = N * itemsize

                    offsets[column] = ptr + reloffset * itemsize

                    ptr += a

                    ff.seek(ptr, 0)
                    b = numpy.fromfile(ff, dtype='i4', count=1)[0]
                    ptr += 4

                    if a != b or b != blocksize:
                        raise IOError("F77 unformatted meta data for `%s` disagrees with true size: starting = %d truth = %d ending = %d" % (column, a, blocksize, b))

                    itemshape = numpy.prod(spec[1])
                    prec = itemsize // itemshape 
                else:
                    offsets[column] = ptr
                    warnings.warn("Cannot decide the item size of `%s`, assuming 4 bytes." % (column))
                    prec = None

                if spec[0] == 'auto':
                    if column == 'ID':
                        mapping = {8:'i8', 4:'i4', None:'i4'}
                    else:
                        mapping = {8:'f8', 4:'f4', None:'f8'}

                    spec = mapping[prec], spec[1]

                if column == "Mass" or ptype in ptypes:
                    dtype.append((column, spec))

                defs.append((column, spec, ptypes))

        dtype = numpy.dtype(dtype)

        self.defs = defs

        BinaryFile.__init__(self, path, dtype=dtype, header_size=256+4+4, offsets=offsets, size=int(header['Npart'][ptype]))


    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the binary file (in particles)

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

        if stop > self.size or start > self.size or start < 0 or stop < 0:
            raise IndexError("start : %d stop %d beyond size of data set %d"
                % (start, stop, self.size))

        dt = [(col, self.dtype[col]) for col in columns]
        toret = numpy.empty(tools.get_slice_size(start, stop, step), dtype=dt)

        with open(self.path, 'rb') as ff:

            for col in columns:
                offset = self.offsets[col]
                dtype = self.dtype[col]
                if col == 'Mass' and self.header_mass != 0:
                    toret[col][:] = self.header_mass
                else:
                    ff.seek(offset, 0)
                    ff.seek(start * dtype.itemsize, 1)
                    toret[col][:] = numpy.fromfile(ff, count=stop-start, dtype=dtype)[::step]

        return toret
