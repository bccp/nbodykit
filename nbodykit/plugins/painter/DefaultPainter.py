from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('DefaultPainter')

class DefaultPainter(Painter):
    plugin_name = "DefaultPainter"

    def __init__(self, weight=None):
        pass
        
    @classmethod
    def register(cls):
        s = cls.schema
        s.add_argument("weight", help="the column giving the weight for each object")

    def paint(self, pm, datasource):
        """
        Paint the ``DataSource`` specified by ``input`` onto the 
        ``ParticleMesh`` specified by ``pm``
    
        Parameters
        ----------
        field : ``DataSource``
            the data source object representing the field to paint onto the mesh
        pm : ``ParticleMesh``
            particle mesh object that does the painting
            
        Returns
        -------
        Ntot : int
            the total number of objects, as determined by painting
        """

        pm.real[:] = 0
        stats = {}

        Nlocal = 0
        if self.weight is None:
            for [position] in datasource.read(['Position']):
                Nlocal += self.basepaint(pm, position)
        else:
            for position, weight in datasource.read(['Position', self.weight]):
                Nlocal += self.basepaint(pm, position, weight)
        stats['Ntot'] = self.comm.allreduce(Nlocal)
        return stats
            
