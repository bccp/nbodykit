from nbodykit.core import Painter
import numpy
from pmesh.pm import RealField

class MomentumPainter(Painter):
    """
    A class to paint the mass-weighted velocity field (momentum) 
    """
    plugin_name = "MomentumPainter"
    
    def __init__(self, velocity_component, moment=1, paintbrush='cic'):
        
        self.velocity_component = velocity_component
        self.moment             = moment
        self.paintbrush         = paintbrush
        
        # initialize the baseclass with the paintbrush
        super(MomentumPainter, self).__init__(paintbrush)
        
        # the index of the velocity component
        self._comp_index = "xyz".index(self.velocity_component)

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "grid the velocity-weighted density field (momentum) field of an " 
        s.description += "input DataSource of objects"
        
        s.add_argument("velocity_component", type=str, choices='xyz',
            help="which velocity component to grid, either 'x', 'y', 'z'")
        s.add_argument("moment", type=int, 
            help="the moment of the velocity field to paint, i.e., "
                 "`moment=1` paints density*velocity, `moment=2` paints density*velocity^2")
        s.add_argument("paintbrush", type=str, help="select a paint brush. Default is to defer to the choice of the algorithm that uses the painter.")
        
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
        real = RealField(pm)
        real[:] = 0
        
        stats = {}
        Nlocal = 0
        
        # open the datasource stream (with no defaults)
        with datasource.open() as stream:
        
            # just paint density as usual
            if self.moment == 0: 
                for [position] in stream.read(['Position']):
                    self.basepaint(real, position, paintbrush=self.paintbrush)
                    Nlocal += len(position)
            # paint density-weighted velocity moments
            else:
                for position, velocity in stream.read(['Position', 'Velocity']):
                    self.basepaint(real, position, weight=velocity[:,self._comp_index]**self.moment, paintbrush=self.paintbrush)
                    Nlocal += len(position)
    
        # total N
        stats['Ntot'] = self.comm.allreduce(Nlocal)
        
        # normalize config-space velocity field by mean number density
        # this is (Nmesh**3 / V) / (Ntot / V)
        norm = pm.Nmesh.prod() / (stats['Ntot'])
        real[:] *= norm

        return real, stats
            
