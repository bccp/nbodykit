from nbodykit.extensionpoints import Algorithm, DataSource
from nbodykit import fof
import logging
import numpy

logger = logging.getLogger('FiberCollisions')

def RaDecDataSource(d):
    source = DataSource.registry.RaDecRedshift
    d['unit_sphere'] = True
    return source.from_config(d)
            

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
        pass
            
    @classmethod
    def register(cls):

        s = cls.schema
        s.description = "the application of fiber collisions to a galaxy survey"
        
        s.add_argument("datasource", type=RaDecDataSource,
            help='`RaDecRedshift DataSource; run `nbkit.py --list-datasources RaDecRedshift` for details')
        s.add_argument("collision_radius", type=float, 
            help="the size of the angular collision radius (in degrees)")
        
    def run(self):
        """
        Compute the FOF collision groups
        """
        from nbodykit import fof
        labels = fof.fof(self.datasource, self.collision_radius, 1, comm=self.comm)
        N = numpy.bincount(labels)
        
        
        
        return labels, N

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


