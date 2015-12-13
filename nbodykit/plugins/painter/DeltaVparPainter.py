from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('DeltaVparPainter')

class DeltaVparPainter(Painter):
    plugin_name = "DeltaVparPainter"

    @classmethod
    def register(kls):
        h = kls.parser
        h.add_argument("los", type=str, help="the line-of-sight, either 'x', 'y', 'z'")

    def paint(self, pm, datasource):

        pm.real[:] = 0
        stats = {}
        los = "xyz".index(self.los)
        
        density = numpy.zeros_like(pm.real)
        momentum = numpy.zeros_like(pm.real)
        
        for i, (position, velocity) in enumerate(self.read_and_decompose(pm, datasource, ['Position', 'Velocity'], stats)):
            
            # paint density first
            pm.real[:] = 0.
            pm.paint(position)
            density[:] += pm.real[:]
            
            # paint momentum
            pm.real[:] = 0.
            pm.paint(position, velocity[:,los])
            momentum[:] += pm.real[:]
            
            
        nonzero = density != 0.
        momentum[nonzero] = momentum[nonzero] / density[nonzero]
        
        density /= density.mean()
        density -= 1.0
        print "mean of density = ", density.mean()
        momentum *= density
        pm.real[:] = momentum[:]
    
        return stats['Ntot']
            
