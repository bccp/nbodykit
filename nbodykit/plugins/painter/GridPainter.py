from nbodykit.extensionpoints import Painter
import numpy
import logging

logger = logging.getLogger('GridPainter')

class GridPainter(Painter):
    """
    Painter that reads a flattened 1D quantity from a binary file, 
    which corresponds to gridded field values on a (Nmesh, Nmesh, Nmesh)
    mesh
    """
    plugin_name = "GridPainter"
    
    def __init__(self):
        pass

    @classmethod
    def register(cls):
        pass
        
    def paint(self, pm, datasource):
        """
        Read the datasource (which specifies the binary file) and
        set ``pm.real`` appropriately given the gridded data
    
        Parameters
        ----------
        pm : ``ParticleMesh``
            particle mesh object that does the painting
        datasource : ``DataSource``
            the data source object representing the field to paint onto the mesh
            
        Returns
        -------
        Ntot : int
            returns ``0`` for the moment, since we don't have info
            on the total number of particles
        """            
        pm.real[:] = 0
        shape = (pm.Nmesh,)*pm.partition.Ndim
        try:
            dtype = numpy.dtype(datasource.dtype)
        except:
            raise ValueError("requested data type `%s` not understood" %datasource.dtype)

        # this gives tuples of integer indices (start, stop) for each dimension
        # that this rank is responsible for 
        slices = [(start, start+width) for start,width in zip(pm.partition.local_i_start, pm.partition.local_ni)]
        
        with open(datasource.path, 'rb') as ff:
        
            for i, a in enumerate(range(*slices[0])):
                for j, b in enumerate(range(*slices[1])):
                    start = numpy.ravel_multi_index([a, b, slices[2][0]], shape)
                    stop = numpy.ravel_multi_index([a, b, slices[2][1]-1], shape)
                    
                    # crash on error - likely Nmesh problem
                    try:
                        ff.seek(start*dtype.itemsize, 0)
                        pm.real[i,j,:] = numpy.fromfile(ff, count=stop-start+1, dtype=dtype)
                    except Exception as e:
                        args = (pm.Nmesh, str(e))
                        raise ValueError("cannot read binary file using `Nmesh = %d`; original message: %s" %args)
                        
        return {}
            