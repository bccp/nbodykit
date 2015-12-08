from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('WeightPainter')

class WeightPainter(Painter):
    field_type = "WeightPainter"

    @classmethod
    def register(kls):
        h = kls.parser
        h.add_argument("-weight", default=None, help="column for the weight")

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

        if self.weight is None:
            for [position] in self.read(pm, datasource, ['Position'], stats):
                pm.paint(position)
        else:
            for position, weight in self.read(pm, datasource, ['Position', self.weight], stats):
                pm.paint(position, weight)

        return stats['Ntot']
            
