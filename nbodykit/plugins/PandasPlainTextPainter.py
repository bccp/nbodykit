from nbodykit.plugins import InputPainter

import numpy
import logging
from nbodykit import files

#------------------------------------------------------------------------------          
class PandasPlainTextPainter(InputPainter):
    field_type = "PandasPlainText"
    
    @classmethod
    def register(kls):
        
        args = kls.field_type+":path:names"
        options = "[:-usecols='x' 'y' 'z'][:-poscols='x' 'y' 'z']\n[:-velcols='vx' 'vy' 'vz']" + \
                  "[:-rsd=[x|y|z]][:-posf=0.001][:-velf=0.001][:-select=flags]"
        h = kls.add_parser(kls.field_type, usage=args+options)
        
        h.add_argument("path", help="path to file")
        h.add_argument("names", nargs='+', type=str, 
            help="names of columns in file")
        h.add_argument("-usecols", nargs='+', type=str, 
            help="only read these columns from file")
        h.add_argument("-poscols", nargs=3, default=['x','y','z'], 
            help="names of the position columns")
        h.add_argument("-velcols", nargs=3, default=None,
            help="names of the velocity columns")
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
        h.add_argument("-posf", default=0.001, type=float, 
            help="factor to scale the positions")
        h.add_argument("-velf", default=0.001, type=float, 
            help="factor to scale the velocities")
        h.add_argument("-select", default=None, type=files.FileSelection, 
            help='row selection based on flags specified as string')
        h.set_defaults(klass=kls)
    
    def paint(self, ns, pm):
        if pm.comm.rank == 0:
            try:
                import pandas as pd
            except:
                raise ImportError("pandas must be installed to use PandasPlainTextPainter")
                
            # read in the plain text file using pandas
            kwargs = {}
            kwargs['comment'] = '#'
            kwargs['names'] = self.names
            kwargs['header'] = None
            kwargs['engine'] = 'c'
            kwargs['delim_whitespace'] = True
            kwargs['usecols'] = self.usecols
            data = pd.read_csv(self.path, **kwargs)
            
            # select based on input flags
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            
            # get position and velocity, if we have it
            pos = data[self.poscols].values
            pos *= self.posf
            if self.velcols is not None:
                vel = data[self.velcols].values
                vel *= self.velf
            else:
                vel = numpy.empty(0, dtype=('f4', 3))
        else:
            pos = numpy.empty(0, dtype=('f4', 3))
            vel = numpy.empty(0, dtype=('f4', 3))

        Ntot = len(pos)
        Ntot = pm.comm.bcast(Ntot)

        # assumed the position values are now in same
        # units as ns.BoxSize
        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= ns.BoxSize # enforce periodic boundary conditions

        layout = pm.decompose(pos)
        tpos = layout.exchange(pos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot

