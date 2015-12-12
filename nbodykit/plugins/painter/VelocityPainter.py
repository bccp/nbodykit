from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('VelocityPainter')

class VelocityPainter(Painter):
    plugin_name = "VelocityPainter"

    @classmethod
    def register(kls):
        h = kls.parser
        h.add_argument("velocity_comp", type=str, help="which velocity component to grid, either 'x', 'y', 'z'")

    def paint(self, pm, datasource):
        """
        Paint the ``DataSource`` specified by ``input`` onto the 
        ``ParticleMesh`` specified by ``pm``
    
        Parameters
        ----------
        pm : ``ParticleMesh``
            particle mesh object that does the painting
        datasource : ``DataSource``
            the data source object representing the field to paint onto the mesh
            
        Returns
        -------
        Ntot : int
            the total number of objects, as determined by painting
        """
        pm.real[:] = 0
        stats = {}
        comp = "xyz".index(self.velocity_comp)
        
        for position, velocity in self.read_and_decompose(pm, datasource, ['Position', 'Velocity'], stats):
            
            # paint density first
            pm.paint(position)
            norm = pm.real.copy()
            
            # paint momentum
            pm.paint(position, velocity[:,comp])
            
            nonzero = norm != 0.
            pm.real[nonzero] = pm.real[nonzero] / norm[nonzero]
    
        return stats['Ntot']
            
