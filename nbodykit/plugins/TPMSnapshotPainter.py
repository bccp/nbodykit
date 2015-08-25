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
        h.set_defaults(klass=kls)

    def read(self, comm):
        Ntot = 0
        columns = ['Position']
        if self.rsd is not None:
            columns.append('Velocity')
            
        for round, P in enumerate(
                files.read(comm, 
                    self.path, 
                    files.TPMSnapshotFile, 
                    columns=columns, 
                    bunchsize=self.bunchsize)):

            if comm.rank == 0:
                logging.info('round %d, nread %d' % (round, len(P['Position'])))

            if self.rsd is not None:
                yield (P['Position'], P['Velocity'], None)
            else:
                yield (P['Position'], None)


#------------------------------------------------------------------------------
