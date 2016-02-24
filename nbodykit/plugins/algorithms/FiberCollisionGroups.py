from nbodykit.extensionpoints import Algorithm, DataSource
from nbodykit import fof
import logging
import numpy

logger = logging.getLogger('FiberCollisions')

class UnitCartesian(DataSource):
    plugin_name = 'UnitCartesian'
        
    def __init__(self, source):
        self.source = source
        self.BoxSize = [1., 1., 1.]
    
    @classmethod
    def register(cls):
        s = cls.schema
        s.add_argument('source', help='the datasource that returns (`RA`, `DEC`)')
    
    def _to_unit_cartesian(self, ra, dec):
        """
        Return the cartesian coordinates on the unit sphere
        """
        x = numpy.cos(ra)*numpy.cos(dec)
        y = numpy.sin(ra)*numpy.cos(dec)
        z = numpy.sin(dec)
        return numpy.vstack([x,y,z]).T
        
    def read(self, columns, stats, full=False):
        if len(columns) > 1 or columns[0] != 'Position':
            raise ValueError("`UnitCartesian` only returns 'Position'")
        
        for [ra, dec] in self.source.read(['RA', 'DEC'], stats, full=full):
            pos = self._to_unit_cartesian(ra, dec)
            yield [pos]
        

class FiberCollisionGroupsAlgorithm(Algorithm):
    """
    Run an angular FOF algorithm to determine fiber collision
    groups from an input catalog, and then determine the
    following population of objects 
    
        * population 1: 
            the "clean" sample of objects in which each object is not 
            angularly collided with any other object in this subsample
        * population 2:
            the potentially-collided objects; these objects are those
            that are fiber collided + those that have been "resolved"
            due to multiple coverage in tile overlap regions
    
    See Guo et al. 2010 (http://arxiv.org/abs/1111.6598)for further details
    """
    plugin_name = "FiberCollisionGroups"
    
    def __init__(self, datasource, collision_radius=62/60./60.):
        
        # create the DataSource that returns cartesian coords on unit sphere
        self.datasource = UnitCartesian(self.datasource)
    
    @classmethod
    def register(cls):
        from nbodykit.extensionpoints import DataSource

        s = cls.schema
        s.description = "the application of fiber collisions to a galaxy survey"
        
        s.add_argument("datasource", type=DataSource.from_config, 
            help='`DataSource` with `RA`, `DEC` columns; run --list-datasources for options')
        s.add_argument("collision_radius", type=float, 
            help="the size of the angular collision radius (in degrees)")
        
    def run(self):
        """
        Compute the FOF collision groups
        """
        labels = fof.fof(self.datasource, self.collision_radius, 1, comm=self.comm)
        Ntot = self.comm.allreduce(len(labels))
        return labels, Ntot

    def save(self, output, data):
        labels, Ntot = data
        print labels, Ntot
        # if self.comm.rank == 0:
        #     with h5py.File(output, 'w') as ff:
        #         # do not create dataset then fill because of
        #         # https://github.com/h5py/h5py/pull/606
        #
        #         dataset = ff.create_dataset(
        #             name='FOFGroups', data=catalog
        #             )
        #         dataset.attrs['Ntot'] = Ntot
        #         dataset.attrs['LinkLength'] = self.linklength
        #         dataset.attrs['BoxSize'] = self.datasource.BoxSize
        #
        # if not self.without_labels:
        #     output = output.replace('.hdf5', '.labels')
        #     bf = bigfile.BigFileMPI(self.comm, output, create=True)
        #     with bf.create_from_array("Label", labels, Nfile=(self.comm.size + 7)// 8) as bb:
        #         bb.attrs['LinkLength'] = self.linklength
        #         bb.attrs['Ntot'] = Ntot
        #         bb.attrs['BoxSize'] = self.datasource.BoxSize
        return


