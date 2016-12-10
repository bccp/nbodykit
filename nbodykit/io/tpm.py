from .binary import BinaryFile

class TPMBinaryFile(BinaryFile):
    """
    Read snapshot files from Martin White's TPM simulations, which
    are stored in a binary format
    """    
    def __init__(self, path, precision='f4'):
        """
        Parameters
        ----------
        path : str
            the path to the binary file to load
        precision : {'f4', 'f8'}, optional
            the string dtype specifying the precision; 
        """
        if precision not in ['f4', 'f8']:
            raise ValueError("precision should be either 'f4' or 'f8'")
            
        dtype = [('Position', (precision, 3)), ('Velocity', (precision, 3)), ('ID', 'u8')]
        BinaryFile.__init__(self, path, dtype=dtype, header_size=28)