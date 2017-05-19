from nbodykit.extern.six import add_metaclass

import abc
import numpy
import logging
from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource

# for converting from particle to mesh
from pmesh import window
from pmesh.pm import RealField, ComplexField

class CatalogMeshSource(MeshSource, CatalogSource):
    logger = logging.getLogger('CatalogMeshSource')
    def __repr__(self):
        return "(%s as CatalogeMeshSource)" % repr(self.source)

    # intended to be used by CatalogSource internally
    def __init__(self, source, BoxSize, Nmesh, dtype, weight, selection, position='Position'):
        # ensure self.comm is set, though usually already set by the child.
        self.comm = source.comm

        self.source    = source
        self.position  = position
        self.selection = selection
        self.weight    = weight

        self.attrs.update(source.attrs)

        # this will override BoxSize and Nmesh carried from the source, if there is any!
        MeshSource.__init__(self, BoxSize=BoxSize, Nmesh=Nmesh, dtype=dtype, comm=source.comm)
        CatalogSource.__init__(self, comm=source.comm)
        
        # copy over the overrides
        self._overrides.update(self.source._overrides)
        
        self.attrs['position'] = self.position
        self.attrs['selection'] = self.selection
        self.attrs['weight'] = self.weight
        self.attrs['compensated'] = True
        self.attrs['interlaced'] = False
        self.attrs['window'] = 'cic'

    @property
    def size(self):
        return self.source.size

    @property
    def hardcolumns (self):
        return self.source.hardcolumns

    def get_hardcolumn(self, col):
        return self.source.get_hardcolumn(col)

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
        # check for 'Position' column
        if self.position not in self:
            raise ValueError("in order to paint a CatalogSource to a RealField, add a " + \
                              "column named '%s', representing the particle positions" %self.position)
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
        columns = [self.position, self.weight, self.selection]
        Position, Weight, Selection = self.read(columns)
        
        # ensure the slices are synced, since decomposition is collective
        N = max(pm.comm.allgather(len(Position)))

        # paint data in chunks on each rank
        chunksize = 1024 ** 2
        for i in range(0, N, chunksize):
            s = slice(i, i + chunksize)

            if len(Position) != 0:
                
                # be sure to use the source to compute
                position, weight, selection = self.source.compute(Position[s], Weight[s], Selection[s])
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
                real.paint(p, mass=w, resampler=paintbrush, hold=True)
            else:
                lay = pm.decompose(position, smoothing=1.0 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)

                H = pm.BoxSize / pm.Nmesh

                # in mesh units
                shifted = pm.affine.shift(0.5)

                real.paint(p, mass=w, resampler=paintbrush, hold=True)
                real2.paint(p, mass=w, resampler=paintbrush, transform=shifted, hold=True)
                c1 = real.r2c()
                c2 = real2.r2c()

                for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                    kH = sum(k[i] * H[i] for i in range(3))
                    s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

                c1.c2r(real)

        N = pm.comm.allreduce(Nlocal)
        nbar = 1.0 * N / numpy.prod(pm.Nmesh)
        
        # make sure we painted something!
        if N == 0:
            raise ValueError("trying to paint particle source to mesh, but no particles were found!")
        shotnoise = numpy.prod(pm.BoxSize) / N

        real.attrs = {}
        real.attrs['shotnoise'] = shotnoise
        real.attrs['N'] = N

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
        actions = MeshSource.actions.fget(self)
        if self.compensated:
            actions = self._get_compensation() + actions
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
        tmp = (numpy.sinc(0.5 * wi / numpy.pi) ) ** 3
        v = v / tmp
    return v

def CompensateCIC(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the CIC window function in configuration space
    
    References
    ----------
    see equation 18 (with p=2) of Jing et al 2005 (arxiv:0409240)
    """     
    for i in range(3):
        wi = w[i]
        tmp = (numpy.sinc(0.5 * wi / numpy.pi) ) ** 2
        tmp[wi == 0.] = 1.
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

