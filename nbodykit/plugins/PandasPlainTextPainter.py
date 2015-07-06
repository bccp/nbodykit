from nbodykit.plugins import InputPainter

import numpy
from nbodykit import files

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
        
        args = kls.field_type+":path:names"
        options = "[:-usecols= x y z][:-poscols= x y z]\n[:-velcols= vx vy vz]" + \
                  "[:-rsd=[x|y|z]][:-posf=0.001][:-velf=0.001][:-select=flags]"
        h = kls.add_parser(kls.field_type, usage=args+options)
        
        h.add_argument("path", help="path to file")
        h.add_argument("names", type=list_str, 
            help="names of columns in file")
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

