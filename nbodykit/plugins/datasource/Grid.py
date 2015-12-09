from nbodykit.extensionpoints import DataSource
import logging

logger = logging.getLogger('Grid')


class GridDataSource(DataSource):
    """
    Class to read field gridded data from a binary file.
    
    Notes
    -----
    * Reading is designed to be done by `GridPainter`, which
      reads gridded quantity straight into the `ParticleMesh`

    
    Parameters
    ----------
    path    : str
        the path of the file to read the data from 
    BoxSize : float
        the box size
    """
    field_type = "Grid"
    
    @classmethod
    def register(kls):
        
        h = kls.parser
        
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=kls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-dtype", default='f4', type=str, help="data type of binary file to read")

