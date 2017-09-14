from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource, CatalogSourceBase
from six import add_metaclass
import abc
import numpy
import logging

# for converting from particle to mesh
from pmesh import window
from pmesh.pm import RealField, ComplexField

class CatalogMesh(CatalogSource, MeshSource):
    """
    A view of a CatalogSource object which knows how to create a MeshSource
    object from itself. The original CatalogSource object is stored as the
    :attr:`source` attribute.

    Parameters
    ----------
    source : CatalogSource
        the input catalog that we are viewing as a mesh
    BoxSize :
        the size of the box
    Nmesh : int, 3-vector
        the number of cells per mesh side
    dtype : str
        the data type of the values stored on mesh
    weight : str
        column in ``source`` that specifies the weight value for each
        particle in the ``source`` to use when gridding
    value : str
        column in ``source`` that specifies the field value for each particle;
        the mesh stores a weighted average of this column
    selection : str
        column in ``source`` that selects the subset of particles to grid
        to the mesh
    position : str, optional
        column in ``source`` specifying the position coordinates; default
        is ``Position``
    """
    logger = logging.getLogger('CatalogMesh')

    def __repr__(self):
        return "(%s as CatalogMesh)" % repr(self.base)

    def __new__(cls, source, BoxSize, Nmesh, dtype, weight,
                    value, selection, position='Position', interlaced=False,
                    compensated=False, window='cic', **kwargs):

        assert isinstance(source, CatalogSourceBase)

        # view the input source (a CatalogSource) as the desired class
        obj = source._view(cls)

        # copy over the necessary meta-data to attrs
        obj.attrs['BoxSize'] = BoxSize
        obj.attrs['Nmesh'] = Nmesh
        obj.attrs['interlaced'] = interlaced
        obj.attrs['compensated'] = compensated
        obj.attrs['window'] = window

        # store others as straight attributes
        obj.dtype = dtype
        obj.weight = weight
        obj.value = value
        obj.selection = selection
        obj.position = position

        # add in the Mesh Source attributes
        MeshSource.__finalize__(obj, obj)

        return obj

    def __init__(self, *args,  **kwargs):
        pass

    def __finalize__(self, obj):

        attrs = getattr(obj, 'attrs', {})
        self.attrs['Nmesh'] = attrs.get('Nmesh', None)
        self.attrs['BoxSize'] = attrs.get('BoxSize', None)
        self.attrs['interlaced'] = attrs.get('interlaced', False)
        self.attrs['compensated'] = attrs.get('compensated', False)
        self.attrs['window'] = attrs.get('window', 'cic')
        self.attrs.update(attrs)

        self.dtype = getattr(obj, 'dtype', 'f4')
        self.weight = getattr(self, 'weight', 'Weight')
        self.position = getattr(self, 'position', 'Position')
        self.value = getattr(self, 'value', 'Value')
        self.selection = getattr(self, 'selection', 'Selection')

        # also initialize the mesh source
        if isinstance(obj, CatalogMesh):
            MeshSource.__finalize__(self, self)

    @property
    def interlaced(self):
        """
        Whether to use interlacing when interpolating the density field.
        See :ref:`the documentation <interlacing>` for further details.

        See also: Section 3.1 of
        `Sefusatti et al. 2015 <https://arxiv.org/abs/1512.07295>`_
        """
        return self.attrs['interlaced']

    @interlaced.setter
    def interlaced(self, interlaced):
        self.attrs['interlaced'] = interlaced

    @property
    def window(self):
        """
        String specifying the name of the interpolation kernel when
        gridding the density field.

        See :ref:`the documentation <window-kernel>` for further details.

        .. note::
            Valid values must be in :attr:`pmesh.window.methods`
        """
        return self.attrs['window']

    @window.setter
    def window(self, value):
        assert value in window.methods
        self.attrs['window'] = value

    @property
    def compensated(self):
        """
        Boolean flag to indicate whether to correct for the windowing
        kernel introduced when interpolating the discrete particles to
        a continuous field.

        See :ref:`the documentation <compensation>` for further details.
        """
        return self.attrs['compensated']

    @compensated.setter
    def compensated(self, value):
        self.attrs['compensated'] = value

    def to_real_field(self, out=None, normalize=True):
        """
        Paint the density field, by interpolating the position column
        on to the mesh.

        This computes the following meta-data attributes in the process of
        painting, returned in the :attr:`attrs` attributes of the returned
        RealField object:

        - N : int
            the (unweighted) total number of objects painted to the mesh
        - W : float
            the weighted number of total objects, equal to the collective
            sum of the 'weight' column
        - shotnoise : float
            the Poisson shot noise, equal to the volume divided by ``N``
        - num_per_cell : float
            the mean number of weighted objects per cell

        .. note::

            The density field on the mesh is normalized as :math:`1+\delta`,
            such that the collective mean of the field is unity.

        See the :ref:`documentation <painting-mesh>` on painting for more
        details on painting catalogs to a mesh.

        Returns
        -------
        real : :class:`pmesh.pm.RealField`
            the painted real field; this has a ``attrs`` dict storing meta-data
        """
        # check for 'Position' column
        if self.position not in self:
            msg = "in order to paint a CatalogSource to a RealField, add a "
            msg += "column named '%s', representing the particle positions" %self.position
            raise ValueError(msg)

        pm = self.pm
        Nlocal = 0 # (unweighted) number of particles read on local rank
        Wlocal = 0 # (weighted) number of particles read on local rank

        # the paint brush window
        paintbrush = window.methods[self.window]

        # initialize the RealField to returns
        if out is not None:
            assert isinstance(out, RealField), "output of to_real_field must be a RealField"
            numpy.testing.assert_array_equal(out.pm.Nmesh, pm.Nmesh)
            real = out
        else:
            real = RealField(pm)
            real[:] = 0

        # need 2nd field if interlacing
        if self.interlaced:
            real2 = RealField(pm)
            real2[...] = 0

        # read the necessary data (as dask arrays)
        columns = [self.position, self.weight, self.value, self.selection]
        Position, Weight, Value, Selection = self.read(columns)

        # perform optimized selection
        sel = self.base.compute(Selection) # compute first, so we avoid repeated computes
        Position = Position[sel]
        Weight = Weight[sel]
        Value = Value[sel]

        # compute
        position, weight, value = self.base.compute(Position, Weight, Value)

        # ensure the slices are synced, since decomposition is collective
        N = max(pm.comm.allgather(len(Position)))

        # paint data in chunks on each rank
        chunksize = 1024 ** 2
        for i in range(0, N, chunksize):
            s = slice(i, i + chunksize)

            if len(Position) != 0:

                # be sure to use the source to compute
                position, weight, value = \
                    self.base.compute(Position[s], Weight[s], Value[s])
            else:
                # workaround a potential dask issue on empty dask arrays
                position = numpy.empty((0, 3), dtype=Position.dtype)
                weight = None
                value = None
                selection = None

            if weight is None:
                weight = numpy.ones(len(position))

            if value is None:
                value = numpy.ones(len(position))

            # track total (selected) number and sum of weights
            Nlocal += len(position)
            Wlocal += weight.sum()

            # no interlacing
            if not self.interlaced:
                lay = pm.decompose(position, smoothing=0.5 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)
                v = lay.exchange(value)
                pm.paint(p, mass=w * v, resampler=paintbrush, hold=True, out=real)

            # interlacing: use 2 meshes separated by 1/2 cell size
            else:
                lay = pm.decompose(position, smoothing=1.0 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)
                v = lay.exchange(value)

                H = pm.BoxSize / pm.Nmesh

                # in mesh units
                shifted = pm.affine.shift(0.5)

                # paint to two shifted meshes
                pm.paint(p, mass=w * v, resampler=paintbrush, hold=True, out=real)
                pm.paint(p, mass=w * v, resampler=paintbrush, transform=shifted, hold=True, out=real2)
                c1 = real.r2c()
                c2 = real2.r2c()

                # and then combine
                for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                    kH = sum(k[i] * H[i] for i in range(3))
                    s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

                c1.c2r(real)

        # unweighted number of objects
        N = pm.comm.allreduce(Nlocal)

        # weighted number of objects
        W = pm.comm.allreduce(Wlocal)

        # weighted number density (objs/cell)
        nbar = 1. * W / numpy.prod(pm.Nmesh)

        # make sure we painted something!
        if N == 0:
            raise ValueError(("trying to paint particle source to mesh, "
                              "but no particles were found!"))

        # shot noise is volume / un-weighted number
        shotnoise = numpy.prod(pm.BoxSize) / N

        # save some meta-data
        real.attrs = {}
        real.attrs['shotnoise'] = shotnoise
        real.attrs['N'] = N
        real.attrs['W'] = W
        real.attrs['num_per_cell'] = nbar

        csum = real.csum()
        if pm.comm.rank == 0:
            self.logger.info("painted %d out of %d objects to mesh" %(N,self.base.csize))
            self.logger.info("mean particles per cell is %g", nbar)
            self.logger.info("sum is %g ", csum)
            self.logger.info("normalized the convention to 1 + delta")

        if normalize:
            if nbar > 0:
                real[...] /= nbar
            else:
                real[...] = 1

        return real

    @property
    def actions(self):
        """
        The actions to apply to the interpolated density field, optionally
        included the compensation correction.
        """
        actions = MeshSource.actions.fget(self)
        if self.compensated:
            actions = self._get_compensation() + actions
        return actions

    def _get_compensation(self):
        """
        Return the compensation function, which corrects for the
        windowing kernel.

        The compensation function is computed as:

        - if ``interlaced = True``:
          - :func:`CompensateCIC` if using CIC window
          - :func:`CompensateTSC` if using TSC window
        - if ``interlaced = False``:
          - :func:`CompensateCICAliasing` if using CIC window
          - :func:`CompensateTSCAliasing` if using TSC window
        """
        if self.interlaced:
            d = {'cic' : self.CompensateCIC,
                 'tsc' : self.CompensateTSC}
        else:
            d = {'cic' : self.CompensateCICAliasing,
                 'tsc' : self.CompensateTSCAliasing}

        if not self.window in d:
            raise ValueError("compensation for window %s is not defined" % self.window)

        filter = d[self.window]

        return [('complex', filter, "circular")]

    @staticmethod
    def CompensateTSC(w, v):
        """
        Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the TSC window function in configuration space.

        .. note::
            see equation 18 (with p=3) of
            `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

        Parameters
        ----------
        w : list of arrays
            the list of "circular" coordinate arrays, ranging from
            :math:`[-\pi, \pi)`.
        v : array_like
            the field array
        """
        for i in range(3):
            wi = w[i]
            tmp = (numpy.sinc(0.5 * wi / numpy.pi) ) ** 3
            v = v / tmp
        return v

    @staticmethod
    def CompensateCIC(w, v):
        """
        Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the CIC window function in configuration space

        .. note::
            see equation 18 (with p=2) of
            `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

        Parameters
        ----------
        w : list of arrays
            the list of "circular" coordinate arrays, ranging from
            :math:`[-\pi, \pi)`.
        v : array_like
            the field array
        """
        for i in range(3):
            wi = w[i]
            tmp = (numpy.sinc(0.5 * wi / numpy.pi) ) ** 2
            tmp[wi == 0.] = 1.
            v = v / tmp
        return v

    @staticmethod
    def CompensateTSCAliasing(w, v):
        """
        Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the TSC window function in configuration space,
        as well as the approximate aliasing correction

        .. note::
            see equation 20 of
            `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

        Parameters
        ----------
        w : list of arrays
            the list of "circular" coordinate arrays, ranging from
            :math:`[-\pi, \pi)`.
        v : array_like
            the field array
        """
        for i in range(3):
            wi = w[i]
            s = numpy.sin(0.5 * wi)**2
            v = v / (1 - s + 2./15 * s**2) ** 0.5
        return v

    @staticmethod
    def CompensateCICAliasing(w, v):
        """
        Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the CIC window function in configuration space,
        as well as the approximate aliasing correction

        .. note::
            see equation 20 of
            `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

        Parameters
        ----------
        w : list of arrays
            the list of "circular" coordinate arrays, ranging from
            :math:`[-\pi, \pi)`.
        v : array_like
            the field array
        """
        for i in range(3):
            wi = w[i]
            v = v / (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
        return v

    def save(self, output, dataset='Field', mode='real'):
        """
        Save the mesh as a :class:`~nbodykit.source.mesh.bigfile.BigFileMesh`
        on disk, either in real or complex space.

        Parameters
        ----------
        output : str
            name of the bigfile file
        dataset : str, optional
            name of the bigfile data set where the field is stored
        mode : str, optional
            real or complex; the form of the field to store
        """
        return MeshSource.save(self, output, dataset=dataset, mode=mode)
