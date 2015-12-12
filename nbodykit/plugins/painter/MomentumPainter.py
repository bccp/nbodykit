from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('MomentumPainter')

class MomentumPainter(Painter):
    plugin_name = "MomentumPainter"

    @classmethod
    def register(kls):
        h = kls.parser
        h.add_argument("velocity_comp", type=str, help="which velocity component to grid, either 'x', 'y', 'z'")
        h.add_argument("-moment", default=1, help="raise velocity to this power")

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
        
        # just paint density as usual
        if self.moment == 0: 
            for [position] in self.read_and_decompose(pm, datasource, ['Position'], stats):
                pm.paint(position)
        # paint density-weighted velocity moments
        else:
            for position, velocity in self.read_and_decompose(pm, datasource, ['Position', 'Velocity'], stats):
                pm.paint(position, velocity[:,comp]**self.moment)
    
        # normalize config-space velocity field by mean number density
        norm = 1.*stats['Ntot']/datasource.BoxSize.prod()
        pm.real[:] /= norm

        return stats['Ntot']
            
