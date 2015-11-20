from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser
import numpy
import logging
from nbodykit.utils import selectionlanguage

logger = logging.getLogger('Pandas')

def list_str(value):
    return value.split()
         
class PandasDataSource(DataSource):
    """
    Class to read field data from a Pandas data file
    and paint the field onto a density grid. 
    File types are guessed from the file name extension, or
    specified via `-ftype` commandline argument.  
    For text files, the data is read using `pandas.read_csv`. 
    For HDF5 files (.hdf5), the data is read using `pandas.read_hdf5`.
    
    Data is stored internally in a `pandas.DataFrame`. 

    Notes
    -----
    * `pandas` must be installed to use
    * data file is assumed to be space-separated
    * commented lines must begin with `#`, with all other lines
    providing data values to be read
    * `names` parameter must be equal to the number of data
    columns, otherwise behavior is undefined
    
    Parameters
    ----------
    path    : str
        the path of the file to read the data from 
    names   : list of str
        For text files, one or more strings specifying the names of the data
        columns. Shape must be equal to number of columns
        in the field, otherwise, behavior is undefined.
        For hdf5 files, the name of the pandas data group.
    BoxSize : float or array_like (3,)
        the box size, either provided as a single float (isotropic)
        or an array of the sizes of the three dimensions
    usecols : list of str, optional
         if not None, only these columns will be read from file
    poscols : list of str, optional
        list of three column names to treat as the position data
    velcols : list of str, optional
        list of three column names to treat as the velociy data
    rsd     : [x|y|z], optional
        direction to do the redshift space distortion
    posf    : float, optional
        multiply the position data by this factor
    velf    : float, optional
        multiply the velocity data by this factor
    select  : str, optional
        string specifying how to select a subset of data, based
        on the column names. For example, if there are columns
        `type` and `mass`, you could specify 
        select= "type == central and mass > 1e14"
    """
    field_type = "Pandas"
    
    @classmethod
    def register(kls):
        
        h = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("names", type=list_str, 
            help="names of columns in text file or name of the data group in hdf5 file")
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
                
        h.add_argument("-usecols", type=list_str, 
            metavar="x y z",
            help="only read these columns from file")
        h.add_argument("-poscols", type=list_str, default=['x','y','z'], 
            metavar="x y z",
            help="names of the position columns")
        h.add_argument("-velcols", type=list_str, default=None,
            metavar="vx vy vz",
            help="names of the velocity columns")
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
        h.add_argument("-posf", default=1., type=float, 
            help="factor to scale the positions")
        h.add_argument("-velf", default=1., type=float, 
            help="factor to scale the velocities")
        h.add_argument("-select", default=None, type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
        h.add_argument("-ftype", default='auto', choices=['hdf5', 'text', 'auto'], 
            help='Format of the Pandas storage container. auto is to guess from the file name.')
    
    def read(self, columns, comm, bunchsize):
        if comm.rank == 0:
            try:
                import pandas as pd
            except:
                raise ImportError("pandas must be installed to use PandasPlainTextDataSource")
                
            if self.ftype == 'auto':
                if self.path.endswith('.hdf5'):
                    self.ftype = 'hdf5'
                else: 
                    self.ftype = 'text'
            if self.ftype == 'hdf5':
                # read in the hdf5 file using pandas
                data = pd.read_hdf(self.path, self.names[0], columns=self.usecols)
            elif self.ftype == 'text':
                # read in the plain text file using pandas
                kwargs = {}
                kwargs['comment'] = '#'
                kwargs['names'] = self.names
                kwargs['header'] = None
                kwargs['engine'] = 'c'
                kwargs['delim_whitespace'] = True
                kwargs['usecols'] = self.usecols
                data = pd.read_csv(self.path, **kwargs)

            nobj = len(data)
            
            # select based on input conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            logger.info("total number of objects selected is %d / %d" % (len(data), nobj))
            
            # get position and velocity, if we have it
            pos = data[self.poscols].values.astype('f4')
            pos *= self.posf
            if self.velcols is not None:
                vel = data[self.velcols].values.astype('f4')
                vel *= self.velf
            else:
                vel = numpy.empty(0, dtype=('f4', 3))
            
        else:
            pos = numpy.empty(0, dtype=('f4', 3))
            vel = numpy.empty(0, dtype=('f4', 3))

        P = {}
        if 'Position' in columns:
            P['Position'] = pos
        if 'Velocity' in columns:
            P['Velocity'] = vel

        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            P['Position'][:, dir] += vel[:, dir]
            P['Position'][:, dir] %= self.BoxSize[dir]

        yield [P[key] if key in P else None for key in columns]

