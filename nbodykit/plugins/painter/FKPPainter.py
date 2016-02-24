from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('FKPPainter')

class FKPPainter(Painter):
    plugin_name = "FKPPainter"

    def __init__(self):
        pass
        
    @classmethod
    def register(cls):
        pass

    def paint(self, pm, datasource):

        # setup
        pm.clear()
        columns = ['Position', 'Weight', 'Nbar']
        stats = {}
        stats['A_ran'] = 0.; stats['A_data'] = 0.
        stats['S_ran'] = 0.; stats['S_data'] = 0.
        
        # paint the randoms
        datasource.set_source('randoms')
        for [position, weight, nbar] in self.read_and_decompose(pm, datasource, columns, stats):
            pm.paint(position, weight)
        
        # copy and store the randoms
        randoms_density = pm.real.copy()
        
        # get the random stats (see equations 13-15 of Beutler et al 2013)
        N_ran = stats.pop('Ntot') # total number 
        A_ran = stats.pop('A') # normalization
        S_ran = stats.pop('S') # shot noise parameter 
        
        # paint the data
        pm.clear()
        datasource.set_source('data')
        for [position, weight, nbar] in self.read_and_decompose(pm, datasource, columns, stats):
            pm.paint(position, weight)
            
        # data stats
        N_data = stats.pop('Ntot') # total number
        A_data = stats.pop('A') # normalization
        S_data = stats.pop('S') # shot noise parameter
        
        # FKP weighted density is n_data - alpha*n_ran
        alpha = 1. * N_data / N_ran
        pm.real[:] -= alpha*randoms_density[:]
        
        # store the metadata
        stats['N_data'] = N_data; stats['N_ran'] = N_ran
        stats['A_data'] = A_data; stats['A_ran'] = A_ran
        stats['S_data'] = S_data; stats['S_ran'] = S_ran
        stats['alpha'] = alpha
        
        stats['A_ran'] *= alpha
        stats['S_ran'] *= alpha**2
        stats['shot_noise'] = (S_ran + S_data)/A_ran # the final shot noise estimate for monopole
        
        return stats
            
