from .binary import BinaryFile

class TPMBinaryFile(BinaryFile):
    """
    Read snapshot binary files from Martin White's TPM simulations.

    These files are stored column-wise with a format, with a
    header of size 28 bytes to begin the file.

    The columns are:

    * Position : 'f4', 'f8' precision
        the position data
    * Velocity : 'f4', 'f8' precision
        the velocity data
    * ID : 'u8' precision
        integers specfiying the particle ID

    Parameters
    ----------
    path : str
        the path to the binary file to load
    precision : {'f4', 'f8'}, optional
        the string dtype specifying the precision

    References
    ----------
    White M., 2002, ApJS, 579, 16
    """
    def __init__(self, path, precision='f4'):

        if precision not in ['f4', 'f8']:
            raise ValueError("precision should be either 'f4' or 'f8'")

        dtype = [('Position', (precision, 3)), ('Velocity', (precision, 3)), ('ID', 'u8')]
        BinaryFile.__init__(self, path, dtype=dtype, header_size=28)
