from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser
from nbodykit.utils.mpilogging import MPILoggerAdapter
import numpy
import logging
         
logger = MPILoggerAdapter(logging.getLogger('BOSSChallengeMock'))

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
    field_type = 'BOSSChallengeMock'
    qpar = 1.0
    qperp = 1.0
    
    def __init__(self, d):
        super(BOSSChallengeMockDataSource, self).__init__(d)
        
        # rescale the box size, if scaled = True
        if self.scaled:
            self.BoxSize[-1] *= self.qpar
            self.BoxSize[0:2] *= self.qperp
    
    @classmethod
    def register(kls):
        h = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        h.add_argument("-scaled", action='store_true', 
            help='rescale the parallel and perp coordinates by the AP factor')
    
    def read(self, columns, comm, bunchsize):
        if comm.rank == 0:
            try:
                import pandas as pd
            except:
                raise ImportError("pandas must be installed to use BOSSChallengeMockDataSource")
                
            # read in the plain text file using pandas
            kwargs = {}
            kwargs['comment'] = '#'
            kwargs['names'] = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            kwargs['header'] = None
            kwargs['engine'] = 'c'
            kwargs['delim_whitespace'] = True
            kwargs['usecols'] = ['x', 'y', 'z']
            data = pd.read_csv(self.path, **kwargs)
            nobj = len(data)
            logger.info("total number of objects read is %d" %nobj)
            
            # get position 
            pos = data[['x', 'y', 'z']].values.astype('f4')
        else:
            pos = numpy.empty(0, dtype=('f4', 3))

        # assumed the position values are now in same
        # units as BoxSize 
        if self.scaled:
            logger.info("multiplying by qperp = %.5f" %self.qperp, on=0)
            logger.info("multiplying by qpar = %.5f" %self.qpar, on=0)

            # scale the coordinates
            pos[:,0:2] *= self.qperp
            pos[:,-1] *= self.qpar

        if 'Velocity' in columns:
            raise KeyError("Velocity is not supported")
        P = {}
        P['Position'] = pos
        P['Mass'] = None

        yield [P[key] for key in columns]
        
class BOSSChallengeBoxADataSource(BOSSChallengeMockDataSource):
    field_type = 'BOSSChallengeBoxA'
    qperp = 0.998753592
    qpar = 0.9975277944
    
class BOSSChallengeBoxBDataSource(BOSSChallengeMockDataSource):
    field_type = 'BOSSChallengeBoxB'
    qperp = 0.998753592
    qpar = 0.9975277944
    
class BOSSChallengeBoxCDataSource(BOSSChallengeMockDataSource):
    field_type = 'BOSSChallengeBoxC'
    qperp = 0.9875682111
    qpar = 0.9751013789
    
class BOSSChallengeBoxDDataSource(BOSSChallengeMockDataSource):
    field_type = 'BOSSChallengeBoxD'
    qperp = 0.9916978595
    qpar = 0.9834483344
    
class BOSSChallengeBoxEDataSource(BOSSChallengeMockDataSource):
    field_type = 'BOSSChallengeBoxE'
    qperp = 0.9916978595
    qpar = 0.9834483344
    
class BOSSChallengeBoxFDataSource(BOSSChallengeMockDataSource):
    field_type = 'BOSSChallengeBoxF'
    qperp = 0.998753592
    qpar = 0.9975277944
    
class BOSSChallengeBoxGDataSource(BOSSChallengeMockDataSource):
    field_type = 'BOSSChallengeBoxG'
    qperp = 0.998753592
    qpar = 0.9975277944


    


