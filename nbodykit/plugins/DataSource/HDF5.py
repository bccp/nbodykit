from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser
import numpy
import logging
from nbodykit.utils import selectionlanguage

logger = logging.getLogger('HDF5')

def list_str(value):
    return value.split()
         
class HDF5DataSource(DataSource):
    """
    Class to read field data from a HDF5 data file

    Notes
    -----
    * `h5py` must be installed to use this data source.
    
    Parameters
    ----------
    path    : str
        the path of the file to read the data from 
    dataset: list of str
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
        Full path to the column of the position vector
    velcols : list of str, optional
        Full path to the column of the velocity vector
    masscols : list of str, optional
        Full path to the column of the mass
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
    field_type = "HDF5"
    
    @classmethod
    def register(kls):
        
        h = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("dataset",  help="name of dataset in HDF5 file")
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions.")
                
        h.add_argument("-poscol", default='Position', 
            help="name of the position column")
        h.add_argument("-velcol", default='Velocity',
            help="name of the velocity column")
        h.add_argument("-masscol", default=None,
            help="name of the mass column, None for unit mass")
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
        h.add_argument("-posf", default=1., type=float, 
            help="factor to scale the positions")
        h.add_argument("-velf", default=1., type=float, 
            help="factor to scale the velocities")
        h.add_argument("-select", default=None, type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
    
    def read(self, columns, comm, bunchsize):
        if comm.rank == 0:
            try:
                import h5py
            except:
                raise ImportError("h5py must be installed to use HDF5 reader")
                
            dataset = h5py.File(self.path, mode='r')[self.dataset]
            data = dataset[...]

            nobj = len(data)
            # select based on input conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            logger.info("total number of objects selected is %d / %d" % (len(data), nobj))
            
            # get position and velocity, if we have it
            pos = data[self.poscol].astype('f4')
            pos *= self.posf
            if self.velcol is not None:
                vel = data[self.velcol].astype('f4')
                vel *= self.velf
            else:
                vel = numpy.zeros(nobj, dtype=('f4', 3))
            if self.masscol is not None:
                mass = data[self.masscol]
        else:
            pos = numpy.empty(0, dtype=('f4', 3))
            vel = numpy.empty(0, dtype=('f4', 3))
            mass = numpy.empty(0, dtype='f4')

        if self.masscol is None:
            mass = None

        P = {}
        if 'Position' in columns:
            P['Position'] = pos
        if 'Velocity' in columns or self.rsd is not None:
            P['Velocity'] = vel
        if 'Mass' in columns:
            P['Mass'] = mass

        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            P['Position'][:, dir] += P['Velocity'][:, dir]
            P['Position'][:, dir] %= self.BoxSize[dir]

        yield [P[key] for key in columns]

