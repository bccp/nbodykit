from nbodykit.plugins import InputPainter, BoxSize_t

import numpy
import logging
from nbodykit.utils import selectionlanguage

def list_str(value):
    return value.split()
         
class PandasPlainTextPainter(InputPainter):
    """
    Class to read field data from a plain text ASCII file
    and paint the field onto a density grid. The data is read
    from file using `pandas.read_csv` and is stored internally in 
    a `pandas.DataFrame`
    
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
    field_type = "PandasPlainText"
    
    @classmethod
    def register(kls):
        
        args = kls.field_type+":path:names:BoxSize"
        options = "[:-usecols= x y z][:-poscols= x y z]\n[:-velcols= vx vy vz]" + \
                  "[:-rsd=[x|y|z]][:-posf=1.0][:-velf=1.0][:-select=conditions]"
        h = kls.add_parser(kls.field_type, usage=args+options)
        
        h.add_argument("path", help="path to file")
        h.add_argument("names", type=list_str, 
            help="names of columns in file")
        h.add_argument("BoxSize", type=BoxSize_t,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
                
        h.add_argument("-usecols", type=list_str, 
            help="only read these columns from file")
        h.add_argument("-poscols", type=list_str, default=['x','y','z'], 
            help="names of the position columns")
        h.add_argument("-velcols", type=list_str, default=None,
            help="names of the velocity columns")
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
        h.add_argument("-posf", default=1., type=float, 
            help="factor to scale the positions")
        h.add_argument("-velf", default=1., type=float, 
            help="factor to scale the velocities")
        h.add_argument("-select", default=None, type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
        h.set_defaults(klass=kls)
    
    def paint(self, pm):
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
            nobj = len(data)
            
            # select based on input conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            logging.info("total number of objects selected is %d / %d" % (len(data), nobj))
            
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

        Ntot = len(pos)
        Ntot = pm.comm.bcast(Ntot)

        # assumed the position values are now in same
        # units as BoxSize
        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            pos[:, dir] += vel[:, dir]
            pos[:, dir] %= self.BoxSize[dir] # enforce periodic boundary conditions

        layout = pm.decompose(pos)
        tpos = layout.exchange(pos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos)) 
        return Ntot

