from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('DivergencePainter')

class DivergencePainter(Painter):
    plugin_name = "DivergencePainter"

    @classmethod
    def register(kls):
        pass

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
        density = numpy.zeros_like(pm.real)
        momentum = numpy.zeros((3,)+pm.real.shape)
        
        columns = ['Position', 'Velocity']
        for i, (position, velocity) in enumerate(self.read_and_decompose(pm, datasource, columns, stats)):
            
            # paint density first
            pm.real[:] = 0.
            pm.paint(position)
            density[:] += pm.real[:]
            
            # paint momentum
            for i in range(3):
                pm.real[:] = 0.
                pm.paint(position, velocity[:,i])
                momentum[i,...] += pm.real[:]
            
        nonzero = density != 0.
        complex_sum = numpy.zeros_like(pm.complex)
        momentum[:,nonzero] = momentum[:,nonzero] / density[nonzero]
        for i in range(3):
            pm.real[:] = momentum[i,...]
            pm.r2c()
            pm.complex *= -1j * pm.k[i]
            complex_sum[:] += pm.complex[:]
                
        pm.complex[:] = complex_sum[:]
        pm.c2r()
    
        return stats['Ntot']
            
