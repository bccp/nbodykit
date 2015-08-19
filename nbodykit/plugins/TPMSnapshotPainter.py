from nbodykit.plugins import InputPainter, BoxSize_t

from nbodykit import files 
import logging

#------------------------------------------------------------------------------
class TPMSnapshotPainter(InputPainter):
    field_type = "TPMSnapshot"
    
    @classmethod
    def register(kls):
        
        args = kls.field_type+":path:BoxSize"
        options = "[:-rsd=[x|y|z]][:-mom=[x|y|z]"
        h = kls.add_parser(kls.field_type, usage=args+options)
        
        
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=BoxSize_t,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
        h.add_argument("-bunchsize", type=int, default=1024*1024*4,
            help='Number of particles to read per rank. A larger number usually means faster IO, but less memory for the FFT mesh')
        h.add_argument("-mom", 
            choices="xyz", default=None, help="paint momentum instead of mass")
        h.set_defaults(klass=kls)

    def paint(self, pm):
        pm.real[:] = 0
        Ntot = 0
        columns = ['Position']
        if self.rsd is not None or self.mom is not None:
            columns.append('Velocity')
            
        for round, P in enumerate(
                files.read(pm.comm, 
                    self.path, 
                    files.TPMSnapshotFile, 
                    columns=columns, 
                    bunchsize=self.bunchsize)):

            nread = pm.comm.allreduce(len(P['Position'])) 

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]
                P['Position'][:, dir] %= 1.0 # enforce periodic boundary conditions

            P['Position'] *= self.BoxSize

            layout = pm.decompose(P['Position'])

            P['Position'] = layout.exchange(P['Position'])
            npaint = pm.comm.allreduce(len(P['Position'])) 

            if self.mom is not None:
                dir = "xyz".index(self.mom)
                weight = 1.0 + P['Velocity'][:, dir].copy()
                del P['Velocity']
                weight = layout.exchange(weight)
            else:
                # uniform mass 
                weight = 1
                
            del layout

            pm.paint(P['Position'], weight)

            del P
            del weight

            if pm.comm.rank == 0:
                logging.info('round %d, npaint %d, nread %d' % (round, npaint, nread))
            Ntot = Ntot + nread
        return Ntot

#------------------------------------------------------------------------------
