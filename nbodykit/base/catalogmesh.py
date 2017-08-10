from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource
from six import add_metaclass
import abc
import numpy
import logging

# for converting from particle to mesh
from pmesh import window
from pmesh.pm import RealField, ComplexField

class CatalogMesh(MeshSource, CatalogSource):
    """
    A class to convert a CatalogSource to a MeshSource, by interpolating
    the position of the discrete particles on to a mesh.

    Parameters
    ----------
    source : CatalogSource
        the input catalog that we wish to interpolate to a mesh
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
        return "(%s as CatalogMesh)" % repr(self.source)

    # intended to be used by CatalogSource internally
    def __init__(self, source, BoxSize, Nmesh, dtype, weight,
                    value, selection, position='Position'):

        # ensure self.comm is set, though usually already set by the child.
        self.comm = source.comm
        self.source = source
        self.position = position
        self.selection = selection
        self.weight = weight
        self.value = value

        self.attrs.update(source.attrs)

        # this will override BoxSize and Nmesh carried from the source, if there is any!
        MeshSource.__init__(self, BoxSize=BoxSize, Nmesh=Nmesh, dtype=dtype, comm=source.comm)
        CatalogSource.__init__(self, comm=source.comm)

        # copy over the overrides
        self._overrides.update(self.source._overrides)

        self.attrs['position'] = self.position
        self.attrs['selection'] = self.selection
        self.attrs['weight'] = self.weight
        self.attrs['value'] = self.value
        self.attrs['compensated'] = True
        self.attrs['interlaced'] = False
        self.attrs['window'] = 'cic'

    @property
    def size(self):
        """
        The number of local particles.
        """
        return self.source.size

    @property
    def hardcolumns (self):
        """
        The names of the hard-coded columns in the source.
        """
        return self.source.hardcolumns

    def get_hardcolumn(self, col):
        """
        Return a hard-coded column
        """
        return self.source.get_hardcolumn(col)

    @property
    def interlaced(self):
        """
        Whether to use interlacing when interpolating the density field.

        See :ref:`the documentation <interlacing>` for further details.

        See also: Section 3.1 of `Sefusatti et al. 2015 <https://arxiv.org/abs/1512.07295>`_
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

        See the :ref:`painting-mesh <documentation>` on painting for more 
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

        fullsize = 0 # track how many were selected out
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

        # ensure the slices are synced, since decomposition is collective
        N = max(pm.comm.allgather(len(Position)))

        # paint data in chunks on each rank
        chunksize = 1024 ** 2
        for i in range(0, N, chunksize):
            s = slice(i, i + chunksize)

            if len(Position) != 0:

                # be sure to use the source to compute
                position, weight, value, selection = \
                    self.source.compute(Position[s], Weight[s], Value[s], Selection[s])
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

            # track all particles, before Selection applied
            fullsize += len(position)

            # apply any Selections
            if selection is not None:
                position = position[selection]
                weight = weight[selection]
                value = value[selection]

            # track total (selected) number and sum of weights
            Nlocal += len(position)
            Wlocal += weight.sum()

            # no interlacing
            if not self.interlaced:
                lay = pm.decompose(position, smoothing=0.5 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)
                v = lay.exchange(value)
                real.paint(p, mass=w * v, resampler=paintbrush, hold=True)

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
                real.paint(p, mass=w * v, resampler=paintbrush, hold=True)
                real2.paint(p, mass=w * v, resampler=paintbrush, transform=shifted, hold=True)
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

        # the full size; should be equal to csize
        fullsize = pm.comm.allreduce(fullsize)

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
            self.logger.info("painted %d out of %d objects to mesh" %(N,fullsize))
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
