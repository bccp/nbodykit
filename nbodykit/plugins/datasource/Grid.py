from nbodykit.extensionpoints import DataSource
import logging

logger = logging.getLogger('Grid')

class GridDataSource(DataSource):
    """
    Class to read field gridded data from a binary file
    
    Notes
    -----
    * Reading is designed to be done by `GridPainter`, which
      reads gridded quantity straight into the `ParticleMesh`
    """
    plugin_name = "Grid"
    
    def __init__(self, path, BoxSize, dtype='f4'):
        pass
    
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "read gridded field data from a binary file"
        
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("dtype", type=str, help="data type of binary file to read")

