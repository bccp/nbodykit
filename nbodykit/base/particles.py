from nbodykit.extern.six import add_metaclass

import abc
import numpy
import logging

# for converting from particle to mesh
from pmesh import window
from pmesh.pm import RealField, ComplexField

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
    
    @property
    def interlaced(self):
        try:
            return self._interlaced
        except AttributeError:
            self._interlaced = False
            return self._interlaced

    @interlaced.setter
    def interlaced(self, interlaced):
        self._interlaced = interlaced

    @property
    def window(self):
        try:
            return self._window
        except AttributeError:
            self._window = 'cic'
            return self._window

    @window.setter
    def window(self, value):
        assert value in window.methods
        self._window = value

    def set_brush(self, window='cic', interlaced=False):
        """
        Set the painter
        """
        self.window = window
        self.interlaced = interlaced
        
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
        pm : pmesh.pm.ParticleMesh
            the particle mesh object
        
        Returns
        -------
        real : pmesh.pm.RealField
            the painted real field
        """
        Nlocal = 0 # number of particles read on local rank
        
        # the paint brush window
        paintbrush = window.methods[self.window]

        # initialize the RealField to returns
        real = RealField(pm)
        real[:] = 0

        # need 2nd field if interlacing
        if self.interlaced:
            real2 = RealField(pm)
            real2[...] = 0

        # read the necessary data (as dask arrays)
        columns = ['Position', 'Weight', 'Selection']
        if not all(col in self for col in columns):
            missing = set(columns) - set(self.columns)
            raise ValueError("self does not contain columns: %s" %str(missing))
        Position, Weight, Selection = self.read(columns)

        # ensure the slices are synced, since decomposition is collective
        N = max(pm.comm.allgather(len(Position)))

        # paint data in chunks on each rank
        chunksize = 1024 ** 2
        for i in range(0, N, chunksize):
            if i > len(Position) : i = len(Position)
            s = slice(i, i + chunksize)
            position, weight, selection = self.compute(Position[s], Weight[s], Selection[s])

            if weight is None:
                weight = numpy.ones(len(position))

            if selection is not None:
                position = position[selection]
                weight   = weight[selection]

            Nlocal += len(position)

            if not self.interlaced:
                lay = pm.decompose(position, smoothing=0.5 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)
                real.paint(position, mass=weight, method=paintbrush, hold=True)
            else:
                lay = pm.decompose(position, smoothing=1.0 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)

                H = pm.BoxSize / pm.Nmesh

                # in mesh units
                shifted = pm.affine.shift(0.5)

                real.paint(position, mass=weight, method=paintbrush, hold=True)
                real2.paint(position, mass=weight, method=paintbrush, transform=shifted, hold=True)
                c1 = real.r2c()
                c2 = real2.r2c()

                for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                    kH = sum(k[i] * H[i] for i in range(3))
                    s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

                c1.c2r(real)
        nbar = pm.comm.allreduce(Nlocal) / numpy.prod(pm.BoxSize)

        if nbar > 0:
            real[...] /= nbar

        real.shotnoise = 1 / nbar

        if pm.comm.rank == 0:
            self.logger.info("mean number density is %g", nbar)
            self.logger.info("normalized the convention to 1 + delta")

        return real
