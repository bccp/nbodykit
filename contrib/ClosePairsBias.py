from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser
import numpy
import logging
from nbodykit.utils import selectionlanguage
from scipy.spatial import cKDTree as KDTree

logger = logging.getLogger('CPB')
from numpy.lib.recfunctions import append_fields

def list_str(value):
    return value.split()
         
class ClosePairBiasing(DataSource):
    """
    Reading in nbodykit hdf5 halo catalogue, and filter the
    results by proximity to massive halos.

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
    """
    field_type = "ClosePairBias"
    
    @classmethod
    def register(kls):
        
        h = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("dataset",  help="name of dataset in HDF5 file")
        h.add_argument("BoxSize", type=BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions.")
        h.add_argument("m0", type=float, help="mass of a particle")
        h.add_argument("massive", type=float, help="log10 of mass of 'massive halo'")
        h.add_argument("nmost", type=int, help="Number of halos to use at most, ordered by proximity")
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
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
            data['Position'] *= self.BoxSize
            data['Velocity'] *= self.BoxSize

            data = append_fields(data, 
                ['Mass', 'LogMass', 'Proximity'],
                [data['Length'] * self.m0,
                 numpy.log10(data['Length'] * self.m0),
                 numpy.zeros(len(data)),
                ]
            )
            massive = data[data['LogMass'] > self.massive]
            if len(massive) == 0: 
                raise ValueError("too few massive halos. decrease 'massive'")

            tree = KDTree(massive['Position'])
            d, i = tree.query(data['Position'])
            data['Proximity'][:] = d


            # select based on input conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]

            logger.info("total number of objects selected is %d / %d" % (len(data), nobj))
            data.sort(order='Proximity')
            data = data[:self.nmost]
            logger.info("Using %d objects", self.nmost)
            
            pos = data['Position']
            vel = data['Velocity']
            mass = data['Mass']
        else:
            pos = numpy.empty(0, dtype=('f4', 3))
            vel = numpy.empty(0, dtype=('f4', 3))
            mass = numpy.empty(0, dtype='f4')

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

