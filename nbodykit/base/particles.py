from nbodykit.extern.six import add_metaclass

import abc
import numpy
import logging
from nbodykit.base.particlemesh import ParticleMeshSource

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

        # if size is already computed update csize
        # otherwise the subclass shall call update_csize explicitly.
        if self.size is not NotImplemented:
            self.update_csize()

        self._overrides = {}

    def to_mesh(self, Nmesh=None, BoxSize=None, dtype='f4'):
        if BoxSize is None:
            try:
                BoxSize = self.attrs['BoxSize']
            except KeyError:
                raise ValueError("BoxSize is not supplied but the particle source does not define one in attrs.")
        if Nmesh is None:
            try:
                Nmesh = self.attrs['Nmesh']
            except KeyError:
                raise ValueError("Nmesh is not supplied but the particle source does not define one in attrs.")

        return ParticleMeshSource(self, Nmesh=Nmesh, BoxSize=BoxSize, dtype=dtype, comm=self.comm)

    def update_csize(self):
        """ set the collective size

            Call this function in __init__ of subclass, 
            after .size is a valid value (not NotImplemented)
        """
        self._csize = self.comm.allreduce(self.size)

        self.logger.debug("local number of particles = %d" % self.size)

        if self.comm.rank == 0:
            self.logger.info("total number of particles = %d" % self.csize)

        import dask.array as da

        self._fallbacks = {
                'Selection': da.ones(self.size, dtype='?', chunks=100000),
                   'Weight': da.ones(self.size, dtype='?', chunks=100000),
                          }

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

    def __len__(self):
        """
        The length of ParticleSource is equal to :attr:`size`; this is the 
        local size of the source on a given rank
        """
        return self.size

    def __contains__(self, col):
        return col in self.columns

    @property
    def columns(self):
        """
        The names of the data fields defined for each particle, including overriden columns and fallback columns
        """
        return sorted(set(list(self.hcolumns) + list(self._overrides) + list(self._fallbacks)))

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
        The number of particles in the source on the local rank.
        """
        return NotImplemented

    @abc.abstractmethod
    def get_column(self, col):
        """
        Return a column from the underlying source or from
        the transformation dictionary
        
        Columns are returned as dask arrays
        """
        pass

    def __getitem__(self, col):
        if col in self._overrides:
            r = self._overrides[col]
        elif col in self.hcolumns:
            r = self.get_column(col)
        elif col in self._fallbacks:
            r = self._fallbacks[col]
        else:
            raise KeyError("column `%s` is not defined in this source" % col)
        if not hasattr(r, 'attrs'):
            r.attrs = {}
        r.save = lambda output, dataset=col: save(r, output, dataset, self.comm)
        return r

    def __setitem__(self, col, value):
        import dask.array as da
        assert len(value) == self.size
        self._overrides[col] = da.from_array(value, chunks=getattr(value, 'chunks', 100000))

    def read(self, columns):
        """
        Return the requested columns as dask arrays
        
        Currently, this returns a dask array holding the total amount
        of data for each rank, divided equally amongst the available ranks
        """
        return [self[col] for col in columns]

    def save_attrs(self, filename, dataset='Header'):
        import bigfile
        with bigfile.BigFileMPI(comm=self.comm, filename=filename, create=True) as ff:
            try:
                header = ff.open(dataset)
            except:
                header = ff.create(dataset)
            with header:
                for key in self.attrs:
                    header.attrs[key] = self.attrs[key]


def save(c, output, dataset, comm):
    """ save a dask array to bigfile output as dataset.

        c.attr is stored as block attrs.
    """
    import bigfile

    csize = comm.allreduce(c.size)
    dtype = c.dtype
    if len(c.shape) > 1:
        dtype = (dtype, c.shape[1:])

    with bigfile.BigFileMPI(comm=comm, filename=output, create=True) as ff:
        with ff.create_from_array(dataset, c) as bb:
            if hasattr(c, 'attrs'): 
                for key in c.attrs:
                    bb.attrs[key] = c.attrs[key]
