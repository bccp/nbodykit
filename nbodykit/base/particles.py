from nbodykit.extern.six import add_metaclass

import abc
import numpy
import logging


@add_metaclass(abc.ABCMeta)
class ParticleSource(object):
    """
    Base class for a source of input particles
    
    This combines the process of reading and painting
    """
    logger = logging.getLogger('ParticleSource')

    # called by the subclasses
    def __init__(self, comm):
        # ensure self.comm is set, though usually already set by the child.

        self.comm = comm

        from .painter import Painter
        self._painter = Painter()

        # set the collective size
        self._csize = self.comm.allreduce(self.size)

        self.logger.debug("local number of particles = %d" % self.size)

        if self.comm.rank == 0:
            self.logger.info("total number of particles = %d" % self.csize)
            self.logger.info("attrs = %s" % self.attrs)

    @staticmethod
    def compute(*args, **kwargs):
        """
        Our version of :func:`dask.compute` that computes
        multiple delayed dask collections at once
        
        This should be called on the return value of :func:`read`
        to converts any dask arrays to numpy arrays
        
        Parameters
        -----------
        args : object
            Any number of objects. If the object is a dask 
            collection, it's computed and the result is returned. 
            Otherwise it's passed through unchanged.
        
        Notes
        -----
        The dask default optimizer induces too many (unnecesarry) 
        IO calls -- we turn this off feature off by default.
        
        Eventually we want our own optimizer probably.
        """
        import dask
        
        # XXX find a better place for this function
        kwargs.setdefault('optimize_graph', False)
        return dask.compute(*args, **kwargs)

    def set_transform(self, *transform, **kwargs):
        """
        Set the transform dictionary
        """
        # the existing dict
        t = self.transform
        
        if len(transform):
            if len(transform) != 1:
                raise ValueError("please supply a dictionary as the single positional argument")
            transform = transform[0]
            if not isinstance(transform, dict):
                raise TypeError("`transform` should be a dictionary of callables")
            
            # update the existing dict
            t.update(transform)
        
        # set any kwargs too
        for k in kwargs:
            t[k] = kwargs[k]
    
    def set_painter(self, painter):
        """
        Set the painter
        """
        self._painter = painter
        
    def __len__(self):
        """
        The length of ParticleSource is equal to :attr:`size`; this is the 
        local size of the source on a given rank
        """
        return self.size
    
    def __contains__(self, col):
        return col in self.columns
        
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
        
    @property
    def transform(self):
        """
        A dictionary of callables that return transform data columns
        """
        try:
            return self._transform
        except AttributeError:
            from nbodykit.transform import DefaultSelection, DefaultWeight
            self._transform = {'Selection':DefaultSelection, 'Weight':DefaultWeight}
            return self._transform
            
    @property
    def painter(self):
        """
        The painter class
        """
        return self._painter
    
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
    def columns(self):
        """
        The names of the data fields defined for each particle, including transformed columns
        """
        return sorted(set(list(self.hcolumns) + list(self.transform)))

    @abc.abstractproperty
    def hcolumns(self):
        """
        The names of the hard data fields defined for each particle.
        hard means it is not a transformed field.
        """
        return []
        
    @property
    def csize(self):
        """
        The collective size of the source, i.e., summed across all ranks
        """
        return self._csize

    @abc.abstractproperty
    def size(self):
        """
        The number of particles in the source on the local rank
        """
        return 0

    @abc.abstractmethod
    def __getitem__(self, col):
        """
        Return a column from the underlying source or from
        the transformation dictionary
        
        Columns are returned as dask arrays
        """
        if col in self.transform:
            return self.transform[col](self)
        else:
            raise KeyError("column `%s` is not a valid column name" %col)

    def read(self, columns):
        """
        Return the requested columns as dask arrays
        
        Currently, this returns a dask array holding the total amount
        of data for each rank, divided equally amongst the available ranks
        """
        return [self[col] for col in columns]

    def paint(self, pm):
        """
        paint : (verb) 
            interpolate the `Position` column to the particle mesh
            specified by ``pm``
        """

        # paint and apply any transformations to the real field
        real = self.painter(self, pm)
        self.painter.transform(self, real)

        return real
