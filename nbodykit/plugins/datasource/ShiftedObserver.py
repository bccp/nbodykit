from nbodykit.extensionpoints import DataSource
import numpy
import logging

logger = logging.getLogger('ShiftedObserved')
         
class ShiftedObserverDataSource(DataSource):
    """
    Class to shift the observer using periodic box data
    and impose redshift-space distortions along the the
    resulting observer's line-of-sight
    """
    plugin_name = "ShiftedObserver"
    
    def __init__(self, datasource, translate, rsd=False):        
        
        # make it an array
        self.translate = numpy.array(self.translate)
    
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "establish an explicit observer (outside the box) for a periodic box"
        
        s.add_argument("datasource", type=DataSource.from_config,
            help="the data to translate in order to impose an explicit observer line-of-sight")
        s.add_argument("translate", type=float, nargs=3,
            help="translate the input data by this vector")
        s.add_argument("rsd", type=bool, 
            help="if `True`, impose redshift distortions along the observer's line-of-sight")

    def read(self, columns, full=False):
        
        # request velocity if we are doing RSD
        if self.rsd and 'Velocity' not in columns:
            columns.append('Velocity')
        
        # read position, redshift, and weights from the stream
        for data in self.datasource.read(columns, full=full):
            
            # translate the cartesian coordinates
            if 'Position' in columns:
                i = columns.index('Position')
                data[i] += self.translate
                
            # add in RSD, along observer LOS
            if self.rsd and 'Position' in columns:
                i = columns.index('Position')
                j = columns.index('Velocity')
                pos = data[i]; vel = data[j]
                
                # the peculiar velocity
                rad = numpy.linalg.norm(pos, axis=-1)
                vpec = (pos*vel).sum(axis=-1) / rad
                
                # shift by the peculiar velocity along LOS
                # assuming vpec is normalized appropriately
                line_of_sight = pos / rad[:,None]
                pos +=  vpec[:,None] * line_of_sight
                            
            yield data
        
        
        

