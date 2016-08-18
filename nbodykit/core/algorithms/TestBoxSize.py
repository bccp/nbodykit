from nbodykit.core import Algorithm, DataSource
from nbodykit.distributedarray import GatherArray
import numpy

class TestBoxSizeAlgorithm(Algorithm):
    """
    A utility algorithm to load a `DataSource` and test 
    if all particles are within the specified BoxSize 
    """
    plugin_name = "TestBoxSize"

    def __init__(self, datasource, BoxSize):
        
        self.datasource = datasource
        self.BoxSize    = BoxSize
        
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "test if all objects in a DataSource fit within a specified BoxSize"

        s.add_argument("datasource", type=DataSource.from_config,
            help="DataSource holding the positions to test")
        s.add_argument("BoxSize", type=DataSource.BoxSizeParser,
            help="the size of the box")
        
                                
    def run(self):
        """
        Run the algorithm, which reads the position coordinates
        from the DataSource and finds any out-of-bounds particles
        """
        # local min/max coords
        local_min_pos = numpy.array([numpy.inf]*3)
        local_max_pos = numpy.array([-numpy.inf]*3)
        
        # read Position
        with self.datasource.open() as stream:
            
            # find global min/max of coords
            for [pos] in stream.read(['Position'], full=False):
                if len(pos):
                    local_min_pos = numpy.minimum(local_min_pos, pos.min(axis=0))
                    local_max_pos = numpy.maximum(local_max_pos, pos.max(axis=0))
                    
            # determine the global min/max
            min_pos = numpy.amin(self.comm.allgather(local_min_pos), axis=0)
            max_pos = numpy.amax(self.comm.allgather(local_max_pos), axis=0)
        
            # and the mean coordinate offset
            self.mean_coordinate_offset = 0.5 * (min_pos + max_pos)
            
            toret = []
            
            # now search for out-of-bounds particles
            for [coords] in stream.read(['Position'], full=False):
                
                pos = coords - self.mean_coordinate_offset
                lim = (pos < -0.5*self.BoxSize)|(pos > 0.5*self.BoxSize)
                idx = lim.any(axis=1)
                
                # the index of each out of bounds particle
                index = idx.nonzero()[0]
                
                # the out-of-bounds boolean 3-vector
                out_of_bounds = lim[idx]
                
                # concat and gather to root
                data = numpy.concatenate([index[:,None], out_of_bounds], axis=1)
                data = GatherArray(data, self.comm, root=0)
                if self.comm.rank == 0:
                    toret.append(data)
                    
            if self.comm.rank == 0:
                toret = numpy.concatenate(toret)
                self.logger.info("%d particles found to be out of range" %len(toret))
                return toret
                
    def save(self, output, result):
        """
        Write out the out-of-bounds particles (done by root)
        """
        # only the master rank writes
        if self.comm.rank == 0:
            
            with open(output, 'wb') as ff:
                
                header = "# %d particles out of bounds\n" %len(result)
                header += "# BoxSize: %s\n" %str(self.BoxSize)
                header += "# mean position vector: %s\n" %str(self.mean_coordinate_offset)
                header += "# column 1: index of out-of-bounds particles (starting at 0)\n"
                header += "# column 2: out of range flag for 'x' dimension (1 if out-of-bounds)\n"
                header += "# column 3: out of range flag for 'y' dimension (1 if out-of-bounds)\n"
                header += "# column 4: out of range flag for 'z' dimension (1 if out-of-bounds)\n"
                ff.write(header.encode())
                numpy.savetxt(ff, result, fmt='%d')

