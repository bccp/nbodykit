from nbodykit.extern.six import add_metaclass
import abc
import numpy
import logging

@add_metaclass(abc.ABCMeta)
class GridSource(object):
    """
    Base class for a source in the form of an input grid
    
    Subclasses must define the :func:`paint` function, which
    is abstract in this class
    """
    logger = logging.getLogger('GridSource')

    # called by the subclasses
    def __init__(self, comm):

        # ensure self.comm is set, though usually already set by the child.
        self.comm = comm
                
        if self.comm.rank == 0:
            self.logger.info("attrs = %s" % self.attrs)
    
    def __len__(self):
        """
        Set the length of a grid source to be 0
        """
        return 0
    
    @property
    def attrs(self):
        """
        Dictionary storing relevant meta-data
        """
        try:
            return self._attrs
        except AttributeError:
            self._attrs = {}
            return self._attrs

    @property
    def BoxSize(self):
        """
        A 3-vector specifying the size of the box for this source
        """
        if 'BoxSize' not in self.attrs:
            raise AttributeError("`BoxSize` has not been set in the `attrs` dict")
            
        BoxSize = numpy.array([1, 1, 1.], dtype='f8')
        BoxSize[:] = self.attrs['BoxSize']
        return BoxSize
        
    @abc.abstractmethod
    def paint(self, pm):
        """
        Fill the particle mesh with the loaded grid file, optionally 
        interpolating if the meshes are not the same size
        """
        pass
        
