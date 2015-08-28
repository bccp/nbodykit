from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser
import numpy
import logging
         
class QPMMockDataSource(DataSource):
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
    qpar = 0.9851209643
    qperp = 0.9925056798
    
    def __init__(self, d):
        super(QPMMockDataSource, self).__init__(d)
        self._BoxSize0 = self.BoxSize.copy()
        
        # rescale the box size, if scaled = True
        if self.scaled:
            if self.rsd is None:
                self.BoxSize *= self.qperp
            else:
                dir = 'xyz'.index(self.rsd)
                for i in [0,1,2]:
                    if i == dir:
                        self.BoxSize[i] *= self.qpar
                    else:
                        self.BoxSize[i] *= self.qperp
        
    
    @classmethod
    def register(kls):
        h = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")

        h.add_argument("-scaled", action='store_true', 
            help='rescale the parallel and perp coordinates by the AP factor')
        h.add_argument("-rsd", choices="xyz",
            help="direction to do redshift distortion")
        h.add_argument("-velf", default=1., type=float, 
            help="factor to scale the velocities")
    
    def read(self, columns, comm):
        if comm.rank == 0:
            try:
                import pandas as pd
            except:
                raise ImportError("pandas must be installed to use QPMMockDataSource")
                
            # read in the plain text file using pandas
            kwargs = {}
            kwargs['comment'] = '#'
            kwargs['names'] = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            kwargs['header'] = None
            kwargs['engine'] = 'c'
            kwargs['delim_whitespace'] = True
            kwargs['usecols'] = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            data = pd.read_csv(self.path, **kwargs)
            nobj = len(data)
            
            logging.info("total number of objects read is %d" %nobj)
            
            # get position 
            pos = data[['x', 'y', 'z']].values.astype('f4')
            vel = data[['vx', 'vy', 'vz']].values.astype('f4')
            vel *= self.velf
        else:
            pos = numpy.empty(0, dtype=('f4', 3))
            vel = numpy.empty(0, dtype=('f4', 3))

        # go to redshift-space and wrap periodically
        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= self._BoxSize0[dir] # enforce periodic boundary conditions
        
        # rescale by AP factor
        if self.scaled:
            if comm.rank == 0:
                logging.info("multiplying by qperp = %.5f" %self.qperp)
 
            # rescale positions and volume
            if self.rsd is None:
                pos *= self.qperp
            else:
                if comm.rank == 0:
                    logging.info("multiplying by qpar = %.5f" %self.qpar)
                for i in [0,1,2]:
                    if i == dir:
                        pos[:,i] *= self.qpar
                    else:
                        pos[:,i] *= self.qperp

        P = {}
        P['Position'] = pos
        P['Velocity'] = vel
        P['Mass'] = None

        yield P

    


