from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('FKPPainter')

class FKPPainter(Painter):
    plugin_name = "FKPPainter"

    @classmethod
    def register(kls):
        pass

    def paint(self, pm, datasource):

        # setup
        columns = ['Position', 'Weight', 'Nbar']
        stats = {}
        
        # paint the randoms
        randoms_density = numpy.zeros_like(pm.real)
        
        datasource.set_source('randoms')
        for [position, weight, nbar] in self.read_and_decompose(pm, datasource, columns, stats):
            pm.paint(position, weight)
            randoms_density[:] += pm.real[:]
            
            # see equations 13-15 of Beutler et al 2013
            stats['A_ran'] = (nbar*weight**2).sum()
            stats['S_ran'] = (weight**2).sum()
            
        Nran = stats.pop('Ntot')
        
        # paint the data
        pm.clear()
        datasource.set_source('data')
        for [position, weight, nbar] in self.read_and_decompose(pm, datasource, columns, stats):
            pm.paint(position, weight)
            
            # see equations 13-15 of Beutler et al 2013
            stats['A_data'] = (nbar*weight**2).sum()
            stats['S_data'] = (weight**2).sum()
            
        Ndata = stats.pop('Ntot')
        
        # FKP weighted density is n_data - alpha*n_ran
        alpha = 1. * Ndata / Nran
        pm.real[:] -= alpha*randoms_density[:]
        
        # store some more metadata
        stats['A_ran'] *= alpha
        stats['S_ran'] *= alpha**2
        stats['N_data'] = Ndata
        stats['N_ran'] = Nran
        stats['alpha'] = alpha
        
        return stats
            
