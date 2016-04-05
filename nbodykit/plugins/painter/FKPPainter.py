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
        A_ran = A_data = 0.
        S_ran = S_data = 0.
        N_ran = N_data = 0

        # compute normalization A and shot noise S
        
        # paint the randoms
        datasource.set_stream('randoms')
        for [position, weight, nbar] in datasource.read(columns):
            Nlocal = self.basepaint(pm, position, weight)
            A_ran += (nbar*weight**2).sum()
            N_ran += Nlocal
            S_ran += (weight**2).sum()
        
        logger.info("A_ran = %f" %A_ran)
        logger.info("nbar = %s" %str(nbar))
        logger.info("weight = %s" %str(weight))
        
        A_ran = self.comm.allreduce(A_ran)
        N_ran = self.comm.allreduce(N_ran)
        S_ran = self.comm.allreduce(S_ran)

        # copy and store the randoms
        randoms_density = pm.real.copy()
        
        # get the random stats (see equations 13-15 of Beutler et al 2013)
                # see equations 13-15 of Beutler et al 2013

        # paint the data
        pm.clear()
        datasource.set_stream('data')
        for [position, weight, nbar] in datasource.read(columns):
            Nlocal = self.basepaint(pm, position, weight)
            A_data += (nbar*weight**2).sum()
            N_data += Nlocal # total number 
            S_data += (weight**2).sum()
            
        A_data = self.comm.allreduce(A_data)
        N_data = self.comm.allreduce(N_data)
        S_data = self.comm.allreduce(S_data)

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
            
