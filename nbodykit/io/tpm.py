from .binary import BinaryFile

class TPMBinaryFile(BinaryFile):
    """
    Read snapshot files from Martin White's TPM simulations, which
    are stored in a binary format
    """
    plugin_name = "FileType.TPM"
    
    def __init__(self, path, precision='f4'):
        
        dtype = [('Position', (precision, 3)), 
                 ('Velocity', (precision, 3)),
                 ('ID', 'u8')]
        BinaryFile.__init__(self, path, dtype=dtype, header_size=28)

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "read binary snapshot files from Martin White's TPM snapshots"

        s.add_argument("path", type=str, 
            help='the name of the file to load')
        s.add_argument("precision", type=str, choices=['f8', 'f4'],
            help='precision of floating point numbers')
