from nbodykit.plugins import InputPainter
import numpy
import logging
         
class QPMMockPainter(InputPainter):
    """
    Class to read data from the DR12 BOSS QPM periodic box 
    mocks, which are stored as a plain text ASCII file, and 
    paint the field onto a density grid. The data is read
    from file using `pandas.read_csv` and is stored internally in 
    a `pandas.DataFrame`
    
    Notes
    -----
    * `pandas` must be installed to use
    * columns are `x`, `y`, `z`, `vx`, `vy`, `vz`
    
    Parameters
    ----------
    path   : str
        the path of the file to read the data from 
    scaled : bool, optional
        rescale the parallel and perp coordinates by the AP factor
    rsd    : [x|y|z], optional
        direction to do the redshift space distortion
    velf   : float, optional
        multiply the velocity data by this factor
    """
    field_type = 'QPMMock'
    qpar = 1.020096
    qperp = 1.027742
    
    @classmethod
    def register(kls):
        
        usage = kls.field_type+":path[:-scaled][:-rsd][:-velf]"
        h = kls.add_parser(kls.field_type, usage=usage)
        
        h.add_argument("path", help="path to file")
        h.add_argument("-scaled", action='store_true', 
            help='rescale the parallel and perp coordinates by the AP factor')
        h.add_argument("-rsd", choices="xyz",
            help="direction to do redshift distortion")
        h.add_argument("-velf", default=1., type=float, 
            help="factor to scale the velocities")
        h.set_defaults(klass=kls)
    
    def paint(self, ns, pm):
        if pm.comm.rank == 0:
            try:
                import pandas as pd
            except:
                raise ImportError("pandas must be installed to use QPMMockPainter")
                
            # read in the plain text file using pandas
            kwargs = {}
            kwargs['comment'] = '#'
            kwargs['names'] = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            kwargs['header'] = None
            kwargs['engine'] = 'c'
            kwargs['delim_whitespace'] = True
            data = pd.read_csv(self.path, **kwargs)
            nobj = len(data)
            
            logging.info("total number of objects read is %d" %nobj)
            
            # get position 
            pos = data[['x', 'y', 'z']].values.astype('f4')
            vel = data[['x', 'y', 'z']].values.astype('f4')
            vel *= self.velf
        else:
            pos = numpy.empty(0, dtype=('f4', 3))
            vel = numpy.empty(0, dtype=('f4', 3))

        Ntot = len(pos)
        Ntot = pm.comm.bcast(Ntot)

        # go to redshift-space
        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= ns.BoxSize # enforce periodic boundary conditions
        
        # rescale by AP factor
        if self.scaled:
            if pm.comm.rank == 0:
                logging.info("multiplying by qperp = %.5f" %self.qperp)
                logging.info("multiplying by qpar = %.5f" %self.qpar)
            if self.rsd is None:
                pos *= self.qperp
            else:
                for i in [0,1,2]:
                    if i == dir:
                        pos[:,i] *= self.qpar
                    else:
                        pos[:,i] *= self.qperp

        layout = pm.decompose(pos)
        tpos = layout.exchange(pos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot


    


