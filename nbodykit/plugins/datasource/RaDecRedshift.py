from nbodykit.extensionpoints import DataSource
from nbodykit.utils import selectionlanguage
import logging
import numpy
import itertools

logger = logging.getLogger('RaDecRedshift')

    
class RaDecRedshiftDataSource(DataSource):
    """
    DataSource designed to handle reading (ra, dec, redshift)
    from a plaintext file, using `pandas.read_csv`
    """
    plugin_name = "RaDecRedshift"
    
    def __init__(self, path, names, 
                    usecols=None, sky_cols=['ra','dec'], z_col='z', 
                    weight_col=None, degrees=False, select=None, bunchsize=4*1024*1024):       
        pass
        
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "read (ra, dec, z) from a plaintext file, using Pandas"
        
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("names", type=str, nargs='+', help="the names of columns in text file")
        s.add_argument("usecols", type=str, nargs='*', help="only read these columns from file")
        s.add_argument("sky_cols", type=str, nargs='*',
            help="names of the columns specifying the sky coordinates")
        s.add_argument("z_col", type=str,
            help="name of the column specifying the redshift coordinate")
        s.add_argument("weight_col", type=str,
            help="name of the column specifying the a weight for each object")
        s.add_argument('degrees', type=bool,
            help='set this flag if the input (ra, dec) are in degrees')
        s.add_argument("select", type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
        s.add_argument("bunchsize", type=int, 
            help="the number of objects to read per rank in a bunch")
                  
    def read(self, columns, stats, full=False):        
        try:
            import pandas as pd
        except:
            name = self.__class__.__name__
            raise ImportError("pandas must be installed to use %s" %name)
        
        bunchsize = self.bunchsize
        if full: bunchsize = None
        
        stats['Ntot'] = 0.

        # read in the plain text file using pandas
        kwargs = {}
        kwargs['comment'] = '#'
        kwargs['names'] = self.names
        kwargs['header'] = None
        kwargs['engine'] = 'c'
        kwargs['delim_whitespace'] = True
        kwargs['usecols'] = self.usecols
        kwargs['chunksize'] = bunchsize
        
        # iterate through in parallel
        data_iter = iter(pd.read_csv(self.path, **kwargs))
        data_iter = itertools.islice(data_iter, self.comm.rank, None, self.comm.size)
        
        stop = 0
        while True:
            
            try:
                data = next(data_iter)
                
                # select based on input conditions
                if self.select is not None:
                    mask = self.select.get_mask(data)
                    data = data[mask]

                # rescale the angles
                if self.degrees:
                    data[self.sky_cols] *= numpy.pi/180.

                # get the (ra, dec, z) coords
                cols = self.sky_cols + [self.z_col]
                pos = data[cols].values.astype('f4')

                # get the weights
                w = numpy.ones(len(pos))
                if self.weight_col is not None:
                    w = data[self.weight_col].values.astype('f4')

                P = {}
                P['Position'] = pos
                P['Weight'] = w

                data = [P[key] for key in columns]
                
            except StopIteration:
                stop = 1
                data = None

            if data is not None:
                shape_and_dtype = [(d.shape, d.dtype) for d in data]
                Ntot = len(data[0])
            else:
                shape_and_dtype = None
                Ntot = None
                
            # check if we are stopping
            stop = self.comm.allreduce(stop)    
            if stop == self.comm.size: break # all ranks are done iterating
                
            shape_and_dtype = self.comm.bcast(shape_and_dtype)
            stats['Ntot'] += self.comm.bcast(Ntot)

            if data is None:
                data = [
                    numpy.empty(0, dtype=(dtype, shape[1:]))
                    for shape,dtype in shape_and_dtype
                ]

            yield data
        
              

