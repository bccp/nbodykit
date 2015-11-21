from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser

import numpy
import logging
from nbodykit.utils import selectionlanguage

logger = logging.getLogger('PlainText')

def list_str(value):
    return value.split()

class PlainTextDataSource(DataSource):
    """
    Class to read field data from a plain text ASCII file
    and paint the field onto a density grid. The data is read
    from file using `numpy.recfromtxt` and store the data in 
    a `numpy.recarray`
    
    Notes
    -----
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
        one or more strings specifying the names of the data
        columns. Shape must be equal to number of columns
        in the field, otherwise, behavior is undefined
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
    field_type = "PlainText"
    
    @classmethod
    def register(kls):
        
        h = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("names", type=list_str, 
            help="names of columns in file")
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
    
    def readall(self, columns):
        # read in the plain text file as a recarray
        kwargs = {}
        kwargs['comments'] = '#'
        kwargs['names'] = self.names
        kwargs['usecols'] = self.usecols
        data = numpy.recfromtxt(self.path, **kwargs)
        nobj = len(data)
        
        # select based on input conditions
        if self.select is not None:
            mask = self.select.get_mask(data)
            data = data[mask]
        logger.info("total number of objects selected is %d / %d" % (len(data), nobj))
        
        # get position and velocity, if we have it
        pos = numpy.vstack(data[k] for k in self.poscols).T.astype('f4')
        pos *= self.posf
        if self.velcols is not None:
            vel = numpy.vstack(data[k] for k in self.velcols).T.astype('f4')
            vel *= self.velf
        else:
            vel = numpy.empty(0, dtype=('f4', 3))

        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= self.BoxSize[dir]

        P = {}
        P['Position'] = pos
        P['Velocity'] = vel
        P['Weight'] = numpy.ones(len(pos))
        P['Mass'] = numpy.ones(len(pos))

        return [P[key] for key in columns]

