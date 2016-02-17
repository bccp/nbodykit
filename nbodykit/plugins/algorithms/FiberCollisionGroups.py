from nbodykit.extensionpoints import Algorithm

import logging
import numpy

logger = logging.getLogger('FiberCollisions')

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
    
    @classmethod
    def register(kls):
        from nbodykit.extensionpoints import DataSource

        p = kls.parser
        p.description = "the application of fiber collisions to a galaxy survey"
        p.add_argument("datasource", type=DataSource.fromstring, 
            help='`DataSource` returning (RA, DEC, Z); run --list-datasource for specifics')
        p.add_argument("collision_radius", type=float, metavar='62/60/60', 
            help="the size of the angular collision radius (in degrees)")

        
    def _to_cartesian(self, ra, dec):
        """
        Return the cartesian coordinates on the unit sphere
        """
        x = numpy.cos(ra)*numpy.cos(dec)
        y = numpy.sin(ra)*numpy.cos(dec)
        z = numpy.sin(dec)
        return numpy.vstack([x,y,z]).T
        
    def run(self):
        """
        Compute the FOF collision group
        """
        
        if self.comm.rank == 0:
            
            # read the data
            stats = {}
            [[Position]] = self.datasource.read(['Position'], stats, full=True)
            
            # (ra,dec) to unit sphere
            ra, dec = Position.T
            cartesian = self._to_cartesian(ra, dec)
        
        catalog, labels = fof.fof(self.datasource, self.linklength, self.nmin, self.comm, return_labels=True)
        Ntot = self.comm.allreduce(len(labels))
        if self.without_labels:
            return catalog, Ntot
        else:
            return catalog, labels, Ntot

    def save(self, output, data):
        if self.without_labels:
            catalog, Ntot = data
        else:
            catalog, labels, Ntot = data

        if self.comm.rank == 0:
            with h5py.File(output, 'w') as ff:
                # do not create dataset then fill because of
                # https://github.com/h5py/h5py/pull/606

                dataset = ff.create_dataset(
                    name='FOFGroups', data=catalog
                    )
                dataset.attrs['Ntot'] = Ntot
                dataset.attrs['LinkLength'] = self.linklength
                dataset.attrs['BoxSize'] = self.datasource.BoxSize

        if not self.without_labels:
            output = output.replace('.hdf5', '.labels')
            bf = bigfile.BigFileMPI(self.comm, output, create=True)
            with bf.create_from_array("Label", labels, Nfile=(self.comm.size + 7)// 8) as bb:
                bb.attrs['LinkLength'] = self.linklength
                bb.attrs['Ntot'] = Ntot
                bb.attrs['BoxSize'] = self.datasource.BoxSize
        return


