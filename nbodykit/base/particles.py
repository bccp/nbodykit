from nbodykit.extern.six import add_metaclass

import abc
import numpy
import logging
from nbodykit.base.grid import GridSource

# for converting from particle to mesh
from pmesh import window
from pmesh.pm import RealField, ComplexField

@add_metaclass(abc.ABCMeta)
class ParticleSource(GridSource):
    """
    Base class for a source of input particles
    
    This combines the process of reading and painting
    """
    logger = logging.getLogger('ParticleSource')

    # called by the subclasses
    def __init__(self, BoxSize, Nmesh, dtype, comm):
        # ensure self.comm is set, though usually already set by the child.

        self.comm = comm

        GridSource.__init__(self, BoxSize=BoxSize, Nmesh=Nmesh, dtype=dtype, comm=comm)

        self.attrs['compensated'] = True
        self.attrs['interlaced'] = False
        self.attrs['window'] = 'cic'

        # if size is already computed update csize
        # otherwise the subclass shall call update_csize explicitly.
        if self.size is not NotImplemented:
            self.update_csize()

    def update_csize(self):
        # set the collective size
        self._csize = self.comm.allreduce(self.size)

        self.logger.debug("local number of particles = %d" % self.size)

        if self.comm.rank == 0:
            self.logger.info("total number of particles = %d" % self.csize)

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
        return self.attrs['interlaced']

    @interlaced.setter
    def interlaced(self, interlaced):
        self.attrs['interlaced'] = interlaced

    @property
    def window(self):
        return self.attrs['window']

    @window.setter
    def window(self, value):
        assert value in window.methods
        self.attrs['window'] = value

    @property
    def compensated(self):
        return self.attrs['compensated']

    @compensated.setter
    def compensated(self, value):
        self.attrs['compensated'] = value

    def __len__(self):
        """
        The length of ParticleSource is equal to :attr:`size`; this is the 
        local size of the source on a given rank
        """
        return self.size

    def __contains__(self, col):
        return col in self.columns

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
    def get_column(self, col):
        """
        Return a column from the underlying source or from
        the transformation dictionary
        
        Columns are returned as dask arrays
        """
        pass

    def __getitem__(self, col):
        cc = ResolveColumn(self, self.transform)
        return cc[col]

    def read(self, columns):
        """
        Return the requested columns as dask arrays
        
        Currently, this returns a dask array holding the total amount
        of data for each rank, divided equally amongst the available ranks
        """
        return [self[col] for col in columns]

    def to_real_field(self):
        """
        paint : (verb) 
            interpolate the `Position` column to the particle mesh
            specified by ``pm``
        
        Returns
        -------
        real : pmesh.pm.RealField
            the painted real field
        """
        pm = self.pm

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
            s = slice(i, i + chunksize)

            if len(Position) != 0:
                position, weight, selection = self.compute(Position[s], Weight[s], Selection[s])
            else:
                # workaround a potential dask issue on empty dask arrays
                position = numpy.empty((0, 3), dtype=Position.dtype)
                weight = None
                selection = None

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
                real.paint(p, mass=w, method=paintbrush, hold=True)
            else:
                lay = pm.decompose(position, smoothing=1.0 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)

                H = pm.BoxSize / pm.Nmesh

                # in mesh units
                shifted = pm.affine.shift(0.5)

                real.paint(p, mass=w, method=paintbrush, hold=True)
                real2.paint(p, mass=w, method=paintbrush, transform=shifted, hold=True)
                c1 = real.r2c()
                c2 = real2.r2c()

                for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                    kH = sum(k[i] * H[i] for i in range(3))
                    s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

                c1.c2r(real)

        nbar = 1.0 * pm.comm.allreduce(Nlocal) / numpy.prod(pm.Nmesh)

        shotnoise = numpy.prod(pm.BoxSize) / pm.comm.allreduce(Nlocal)

        real.attrs = {}
        real.attrs['shotnoise'] = shotnoise
        csum = real.csum()
        if pm.comm.rank == 0:
            self.logger.info("mean particles per cell is %g", nbar)
            self.logger.info("sum is %g ", csum)
            self.logger.info("normalized the convention to 1 + delta")

        if nbar > 0:
            real[...] /= nbar
        else:
            real[...] = 1

        return real

    @property
    def actions(self):
        actions = GridSource.actions.fget(self)
        if self.compensated:
            return self._get_compensation() + actions
        return actions

    def _get_compensation(self):
        if self.interlaced:
            d = {'cic' : CompensateCIC,
                 'tsc' : CompensateTSC}
        else:
            d = {'cic' : CompensateCICAliasing,
                 'tsc' : CompensateTSCAliasing}

        if not self.window in d:
            raise ValueError("compensation for window %s is not defined" % self.window)

        filter = d[self.window]

        return [('complex', filter, "circular")]

def CompensateTSC(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the TSC window function in configuration space
    
    References
    ----------
    see equation 18 (with p=3) of Jing et al 2005 (arxiv:0409240)
    """ 
    for i in range(3):
        wi = w[i]
        tmp = ( numpy.sin(0.5 * wi) / (0.5 * wi) ) ** 3
        tmp[k[i] == 0.] = 1.
        v = v / tmp
    return v

def CompensateCIC(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the CIC window function in configuration space
    
    References
    ----------
    see equation 18 (with p=3) of Jing et al 2005 (arxiv:0409240)
    """     
    for i in range(3):
        wi = w[i]
        tmp = ( numpy.sin(0.5 * wi) / (0.5 * wi) ) ** 2
        tmp[kk[i] == 0.] = 1.
        v = v / tmp
    return v

def CompensateTSCAliasing(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the TSC window function in configuration space,
    as well as the approximate aliasing correction

    References
    ----------
    see equation 20 of Jing et al 2005 (arxiv:0409240)
    """   
    for i in range(3):
        wi = w[i]
        s = numpy.sin(0.5 * wi)**2
        v = v / (1 - s + 2./15 * s**2) ** 0.5
    return v

def CompensateCICAliasing(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the CIC window function in configuration space,
    as well as the approximate aliasing correction

    References
    ----------
    see equation 20 of Jing et al 2005 (arxiv:0409240)
    """     
    for i in range(3):
        wi = w[i]
        v = v / (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
    return v

class ResolveColumn(object):
    """ Helper object that provides the context for evaluating
        the transforms.

        As we go along we remove edges from the dictionary to
        avoid deadloops.

        this object is passed to the transform functions,
        so we make it look like a Source.

    """
    def __init__(self, source, transforms):
        self.source = source
        self.transforms = {}
        self.transforms.update(transforms)

    def __len__(self):
        """ len is used some times """
        return len(self.source)

    def __getitem__(self, col):
        if col in self.transforms:
            t = self.transforms.pop(col)
            return t(self)
        else:
            return self.source.get_column(col)
