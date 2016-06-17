from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('MomentumPainter')

class MomentumPainter(Painter):
    plugin_name = "MomentumPainter"
    
    def __init__(self, velocity_component, moment=1):
        self._comp_index = "xyz".index(self.velocity_component)

    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "grid the velocity-weighted density field (momentum) field of an " 
        s.description += "input DataSource of objects"
        
        s.add_argument("velocity_component", type=str, choices='xyz',
            help="which velocity component to grid, either 'x', 'y', 'z'")
        s.add_argument("moment", type=int, 
            help="the moment of the velocity field to paint, i.e., "
                 "`moment=1` paints density*velocity, `moment=2` paints density*velocity^2")

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
        stats : dict
            dictionary of statistics, usually only containing `Ntot`
        """
        pm.clear()
        stats = {}
        Nlocal = 0
        
        # open the datasource stream (with no defaults)
        with datasource.open() as stream:
        
            # just paint density as usual
            if self.moment == 0: 
                for [position] in stream.read(['Position']):
                    Nlocal += self.basepaint(pm, position)
            # paint density-weighted velocity moments
            else:
                for position, velocity in stream.read(['Position', 'Velocity']):
                    Nlocal += self.basepaint(pm, position, velocity[:,self._comp_index]**self.moment)
    
        # total N
        stats['Ntot'] = self.comm.allreduce(Nlocal)
        
        # normalize config-space velocity field by mean number density
        norm = 1.*stats['Ntot']/pm.BoxSize.prod()
        pm.real[:] /= norm

        return stats
            
