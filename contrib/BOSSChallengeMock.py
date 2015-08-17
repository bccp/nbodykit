from nbodykit.plugins import InputPainter

import numpy
import logging
         
class BOSSChallengeMockPainter(InputPainter):
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
    * data is assumed to be in redshift-space, with `z` giving
    the LOS axis
    
    Parameters
    ----------
    path    : str
        the path of the file to read the data from 
    """
    field_type = 'BOSSChallengeMock'
    qpar = 1.0
    qperp = 1.0
    
    @classmethod
    def register(kls):
        
        usage = kls.field_type+":path[:-scaled]"
        h = kls.add_parser(kls.field_type, usage=usage)
        
        h.add_argument("path", help="path to file")
        h.add_argument("-scaled", action='store_true', 
            help='rescale the parallel and perp coordinates by the AP factor')
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
            kwargs['names'] = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            kwargs['header'] = None
            kwargs['engine'] = 'c'
            kwargs['delim_whitespace'] = True
            kwargs['usecols'] = ['x', 'y', 'z']
            data = pd.read_csv(self.path, **kwargs)
            nobj = len(data)
            
            logging.info("total number of objects read is %d" %nobj)
            
            # get position 
            pos = data[['x', 'y', 'z']].values.astype('f4')
        else:
            pos = numpy.empty(0, dtype=('f4', 3))

        Ntot = len(pos)
        Ntot = pm.comm.bcast(Ntot)

        # assumed the position values are now in same
        # units as ns.BoxSize
        if self.scaled:
            if pm.comm.rank == 0:
                logging.info("multiplying by qperp = %.5f" %self.qperp)
                logging.info("multiplying by qpar = %.5f" %self.qpar)
            pos[:,0] *= self.qperp
            pos[:,1] *= self.qperp
            pos[:,2] *= self.qpar

        layout = pm.decompose(pos)
        tpos = layout.exchange(pos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot
        
class BOSSChallengeBoxAPainter(BOSSChallengeMockPainter):
    field_type = 'BOSSChallengeBoxA'
    qperp = 0.998753592
    qpar = 0.9975277944
    
class BOSSChallengeBoxBPainter(BOSSChallengeMockPainter):
    field_type = 'BOSSChallengeBoxB'
    qperp = 0.9875682111
    qpar = 0.9751013789
    
class BOSSChallengeBoxCPainter(BOSSChallengeMockPainter):
    field_type = 'BOSSChallengeBoxC'
    qperp = 0.9875682111
    qpar = 0.9751013789
    
class BOSSChallengeBoxDPainter(BOSSChallengeMockPainter):
    field_type = 'BOSSChallengeBoxD'
    qperp = 0.9916978595
    qpar = 0.9834483344
    
class BOSSChallengeBoxEPainter(BOSSChallengeMockPainter):
    field_type = 'BOSSChallengeBoxE'
    qperp = 0.9916978595
    qpar = 0.9834483344
    
class BOSSChallengeBoxFPainter(BOSSChallengeMockPainter):
    field_type = 'BOSSChallengeBoxF'
    qperp = 0.998753592
    qpar = 0.9975277944
    
class BOSSChallengeBoxGPainter(BOSSChallengeMockPainter):
    field_type = 'BOSSChallengeBoxG'
    qperp = 0.998753592
    qpar = 0.9975277944


    


