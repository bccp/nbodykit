from nbodykit.extensionpoints import DataSource
import numpy
from nbodykit.utils import selectionlanguage
from scipy.spatial import cKDTree as KDTree
import mpsort

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
    plugin_name = "ClosePairBias"
    
    def __init__(self, path, dataset, BoxSize, m0, massive, 
                    rsd=None, select1=None, select2=None):
        pass
    
    @classmethod
    def register(cls):
        
        s = cls.schema
        
        s.add_argument("path", help="path to file")
        s.add_argument("dataset",  help="name of dataset in HDF5 file")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions.")
        s.add_argument("m0", type=float, help="mass of a particle")
        s.add_argument("massive", type=selectionlanguage.Query, 
            help="selection that defines the 'massive halo'; it can also be 'less massive halo' ")
        s.add_argument("sd", choices="xyz", help="direction to do redshift distortion")
        s.add_argument("select1", type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
        s.add_argument("select2", type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
    
    def parallel_read(self, columns, full=False):
        if self.comm.rank == 0:
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

            massive = data[self.massive.get_mask(data)]

            self.logger.info("Selected %d 'massive halos'" % len(massive))
            if len(massive) == 0: 
                raise ValueError("too few massive halos. Check the 'massive' selection clause.")
            
            data = numpy.array_split(data, self.comm.size)
        else:
            massive = None
            data = None

        if self.comm.rank == 0:
            self.logger.info("load balancing ")
        data = self.comm.scatter(data)
        massive = self.comm.bcast(massive)

        if self.comm.rank == 0:
            self.logger.info("Querying KDTree")
        tree = KDTree(massive['Position'])

        nobjs = self.comm.allreduce(len(data))
        if self.comm.rank == 0:
            self.logger.info("total number of objects is %d" % nobjs)

        # select based on input conditions
        if self.select1 is not None:
            mask = self.select1.get_mask(data)
            data = data[mask]
            nobjs1 = self.comm.allreduce(len(data))
            if self.comm.rank == 0:
                self.logger.info("selected (1) number of objects is %d" % (nobjs1 ))

        d, i = tree.query(data['Position'], k=2)

        d[d == 0] = numpy.inf
        data['Proximity'][:] = d.min(axis=-1)

        if len(data) > 0:
            mymax = data['Proximity'].max()
        else:
            mymax = 0
        pbins = numpy.linspace(0, numpy.max(self.comm.allgather(mymax)), 10)
        h = self.comm.allreduce(numpy.histogram(data['Proximity'], bins=pbins)[0])

        if self.comm.rank == 0:
            for p1, p2, h in zip(list(pbins), list(pbins[1:]) + [numpy.inf], h):
                self.logger.info("Proximity: [%g - %g] Halos %d" % (p1, p2, h))

        if self.select2 is not None:
            mask = self.select2.get_mask(data)
            data = data[mask]
            nobjs2 = self.comm.allreduce(len(data))
            if self.comm.rank == 0:
                self.logger.info("selected (2) number of objects is %d (%g %%)" % (nobjs2, 100.0 * nobjs2 / nobjs1))

        meanmass = self.comm.allreduce(data['Mass'].sum(dtype='f8')) \
                 / self.comm.allreduce(len(data))

        if self.comm.rank == 0:
            self.logger.info("mean mass of selected objects is %g (log10 = %g)" 
                % (meanmass, numpy.log10(meanmass)))

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

        P['Weight'] = numpy.ones(len(pos))

        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            P['Position'][:, dir] += P['Velocity'][:, dir]
            P['Position'][:, dir] %= self.BoxSize[dir]

        yield [P.get(key, None) for key in columns]

