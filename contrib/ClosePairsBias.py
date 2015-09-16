from nbodykit.plugins import DataSource
from nbodykit.utils.pluginargparse import BoxSizeParser
import numpy
import logging
from nbodykit.utils import selectionlanguage
from nbodykit.utils.mpilogging import MPILoggerAdapter
from scipy.spatial import cKDTree as KDTree
import mpsort

logger = MPILoggerAdapter(logging.getLogger('CPB'))

def append_fields(data, dict):
    def guessdtype(data):
        return (data.dtype, data.shape[1:])

    names1 = data.dtype.names
    names2 = [name for name in dict]

    dtype = [(name, guessdtype(data[name])) for name in data.dtype.names] \
         +  [(name, guessdtype(dict[name])) for name in dict]
    newdata = numpy.empty(len(data), dtype=dtype)

    for name in data.dtype.names:
        newdata[name] = data[name]
    for name in dict:
        newdata[name] = dict[name]
    return newdata
 
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
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
        h.add_argument("-select1", default=None, type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
        h.add_argument("-select2", default=None, type=selectionlanguage.Query, 
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
                dict(Mass=data['Length'] * self.m0,
                 LogMass=numpy.log10(data['Length'] * self.m0),
                 Proximity=numpy.zeros(len(data)))
            )

            massive = data[data['LogMass'] > self.massive]

            if len(massive) == 0: 
                raise ValueError("too few massive halos. decrease 'massive'")
            
            data = numpy.array_split(data, comm.size)
        else:
            massive = None
            data = None

        logger.info("load balancing ", on=0)
        data = comm.scatter(data)
        massive = comm.bcast(massive)

        logger.info("Querying KDTree", on=0)
        tree = KDTree(massive['Position'])

        nobjs = comm.allreduce(len(data))
        logger.info("total number of objects is %d" % nobjs, on=0)

        # select based on input conditions
        if self.select1 is not None:
            mask = self.select1.get_mask(data)
            data = data[mask]
            nobjs1 = comm.allreduce(len(data))
            logger.info("selected (1) number of objects is %d" % nobjs1, on=0)

        d, i = tree.query(data['Position'])
        data['Proximity'][:] = d

        if len(data) > 0:
            mymax = data['Proximity'].max()
        else:
            mymax = 0
        pbins = numpy.linspace(0, numpy.max(comm.allgather(mymax)), 10)
        h = comm.allreduce(numpy.histogram(data['Proximity'], bins=pbins)[0])

        for p1, p2, h in zip(list(pbins), list(pbins[1:]) + [numpy.inf], h):
            logger.info("Proximity: [%g - %g] Halos %d" % (p1, p2, h), on=0)

        if self.select2 is not None:
            mask = self.select2.get_mask(data)
            data = data[mask]
            nobjs2 = comm.allreduce(len(data))
            logger.info("selected (2) number of objects is %d (%g %%)" % (nobjs2, 100.0 * nobjs2 / nobjs1), on=0)

        meanmass = comm.allreduce(data['Mass'].sum(dtype='f8')) \
                 / comm.allreduce(len(data))

        logger.info("mean mass of selected objects is %g (log10 = %g)" 
                % (meanmass, numpy.log10(meanmass)), on=0)

        pos = data['Position']
        vel = data['Velocity']

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

