import pandas as pd
from nbodykit.extensionpoints import DataSource
import numpy


class BOSSChallengeMockDataSource(DataSource):
    """
    Class to read data from the DR12 BOSS periodic box challenge 
    mocks, which are stored as a plain text ASCII file, and 
    paint the field onto a density grid. The data is read
    from file using `pandas.read_csv` and is stored internally in 
    a `pandas.DataFrame`
    
    Notes
    -----
    * `pandas` must be installed to use
    * first three columns are `x`, `y`, `z`
    * data is assumed to be in redshift-space, with `z` (last axis) 
    giving the LOS axis
    
    Parameters
    ----------
    path    : str
        the path of the file to read the data from
    BoxSize : float or array_like (3,)
        the box size, either provided as a single float (isotropic)
        or an array of the sizes of the three dimensions 
    scaled : bool, optional
        rescale the parallel and perp coordinates by the AP factor
    """
    plugin_name = 'BOSSChallengeMock'
    qpar = 1.0
    qperp = 1.0
    
    def __init__(self, path, BoxSize, scaled=False):
        
        # rescale the box size, if scaled = True
        if self.scaled:
            self.BoxSize[-1] *= self.qpar
            self.BoxSize[0:2] *= self.qperp
    
    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "read from BOSS DR12 challenge mocks"
        
        s.add_argument("path", help="path to file")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("scaled", type=bool,
            help='rescale the parallel and perp coordinates by the AP factor')
    
    def readall(self):
                
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
        self.logger.info("total number of objects read is %d" %nobj)
        
        # get position 
        pos = data[['x', 'y', 'z']].values.astype('f4')

        # assumed the position values are now in same
        # units as BoxSize 
        if self.scaled:
            self.logger.info("multiplying by qperp = %.5f" %self.qperp)
            self.logger.info("multiplying by qpar = %.5f" %self.qpar)

            # scale the coordinates
            pos[:,0:2] *= self.qperp
            pos[:,-1] *= self.qpar

        P = {}
        P['Position'] = pos
        return P
        
class BOSSChallengeBoxADataSource(BOSSChallengeMockDataSource):
    plugin_name = 'BOSSChallengeBoxA'
    qperp = 0.998753592
    qpar = 0.9975277944
    
    def __init__(self, path, BoxSize, scaled=False):
        super(BOSSChallengeBoxADataSource, self).__init__(path, BoxSize, scaled)
    
class BOSSChallengeBoxBDataSource(BOSSChallengeMockDataSource):
    plugin_name = 'BOSSChallengeBoxB'
    qperp = 0.998753592
    qpar = 0.9975277944
    
    def __init__(self, path, BoxSize, scaled=False):
        super(BOSSChallengeBoxBDataSource, self).__init__(path, BoxSize, scaled)
    
class BOSSChallengeBoxCDataSource(BOSSChallengeMockDataSource):
    plugin_name = 'BOSSChallengeBoxC'
    qperp = 0.9875682111
    qpar = 0.9751013789
    
    def __init__(self, path, BoxSize, scaled=False):
        super(BOSSChallengeBoxCDataSource, self).__init__(path, BoxSize, scaled)
    
class BOSSChallengeBoxDDataSource(BOSSChallengeMockDataSource):
    plugin_name = 'BOSSChallengeBoxD'
    qperp = 0.9916978595
    qpar = 0.9834483344
    
    def __init__(self, path, BoxSize, scaled=False):
        super(BOSSChallengeBoxDDataSource, self).__init__(path, BoxSize, scaled)
    
class BOSSChallengeBoxEDataSource(BOSSChallengeMockDataSource):
    plugin_name = 'BOSSChallengeBoxE'
    qperp = 0.9916978595
    qpar = 0.9834483344
    
    def __init__(self, path, BoxSize, scaled=False):
        super(BOSSChallengeBoxEDataSource, self).__init__(path, BoxSize, scaled)
    
class BOSSChallengeBoxFDataSource(BOSSChallengeMockDataSource):
    plugin_name = 'BOSSChallengeBoxF'
    qperp = 0.998753592
    qpar = 0.9975277944
    
    def __init__(self, path, BoxSize, scaled=False):
        super(BOSSChallengeBoxFDataSource, self).__init__(path, BoxSize, scaled)
    
class BOSSChallengeBoxGDataSource(BOSSChallengeMockDataSource):
    plugin_name = 'BOSSChallengeBoxG'
    qperp = 0.998753592
    qpar = 0.9975277944
    
    def __init__(self, path, BoxSize, scaled=False):
        super(BOSSChallengeBoxGDataSource, self).__init__(path, BoxSize, scaled)


class BOSSChallengeNSeriesDataSource(DataSource):
    """
    N-series BOSS challenge mock
    """
    plugin_name = 'BOSSChallengeNSeries'
    qperp = 0.99169902
    qpar = 0.98345263
    
    def __init__(self, path, BoxSize, scaled=False, rsd=None, velf=1.):
        
        # create a copy of the original, before scaling
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
    def register(cls):
        s = cls.schema
        s.description = 'read the BOSS DR12 N-series cutsky mocks'
        
        s.add_argument("path", help="path to file")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("scaled", type=bool,
            help='rescale the parallel and perp coordinates by the AP factor')
        s.add_argument("rsd", choices="xyz", help="direction to do redshift distortion")
        s.add_argument("velf", type=float, help="factor to scale the velocities")
    
    def readall(self):
            
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
        
        self.logger.info("total number of objects read is %d" %nobj)
        
        # get position 
        pos = data[['x', 'y', 'z']].values.astype('f4')
        vel = data[['vx', 'vy', 'vz']].values.astype('f4')
        vel *= self.velf
        
        # go to redshift-space and wrap periodically
        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= self._BoxSize0[dir] # enforce periodic boundary conditions
        
        # rescale by AP factor
        if self.scaled:
            self.logger.info("multiplying by qperp = %.5f" %self.qperp)
 
            # rescale positions and volume
            if self.rsd is None:
                pos *= self.qperp
            else:
                self.logger.info("multiplying by qpar = %.5f" %self.qpar)
                for i in [0,1,2]:
                    if i == dir:
                        pos[:,i] *= self.qpar
                    else:
                        pos[:,i] *= self.qperp

        P = {}
        P['Position'] = pos
        P['Velocity'] = vel
        return P