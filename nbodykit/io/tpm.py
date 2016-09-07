from .binary import BinaryFile

class TPMBinaryFile(BinaryFile):
    plugin_name = "FileType.TPM"
    def __init__(self, path, precision='f4'):
        BinaryFile.__init__(self, path, dtype=[
            ('Position', (precision, 3)),
            ('Velocity', (precision, 3)),
            ('ID', 'u8')], header_size=28)

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "a binary file reader"

        s.add_argument("path", type=str, 
            help='the name of the binary file to load')

        s.add_argument("precision", type=str, choices=['f8', 'f4'],
            help='precision of floating point numbers.')
