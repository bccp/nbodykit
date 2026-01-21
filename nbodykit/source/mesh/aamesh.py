from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource, CatalogSourceBase
from nbodykit import _global_options
import numpy
import logging
import warnings

# for converting from particle to mesh
from pmesh import window
from pmesh.pm import RealField, ComplexField

class SSAAMesh(MeshSource):
    """
    SSAAMesh from a Catalog. SSAA stands for supersampling antialiasing.

    The original CatalogSource object is stored as the :attr:`base` attribute.

    Parameters
    ----------
    source : CatalogSource
        the input catalog that we are viewing as a mesh
    Nmesh : int, 3-vector
        the number of cells per mesh side
    BoxSize :
        the size of the box; None to read from the source.attrs
    dtype : str
        the data type of the values stored on mesh, 'f4'
    samples : array_like
        points to sample.
        `SSAAMesh.GRID` : a uniform grid of 2x2x2
        `SSAA.Mesh.INTERLACE` : interlacing, see Sefusatti et al. 2015;
        also known as Quincunx.
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
    window : str, optional
        the string specifying which window interpolation scheme to use;
        see ``pmesh.window.methods``
    """
    logger = logging.getLogger('AntiAliasingMesh')

    GRID = numpy.array([
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
            (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]) * 0.5

    # also known as Quincunx or HRAA
    # according to https://en.wikipedia.org/wiki/Supersampling
    INTERLACE = numpy.array([
            (0, 0, 0), (1, 1, 1)]) * 0.5

    def __repr__(self):
        return "(%s as AntiAliasingMesh)" % repr(self.base)

    def __init__(self, source, Nmesh, weight='Weight',
                    value='Value',
                    selection='Selection',
                    position='Position', 
                    dtype='f8',
                    BoxSize=None,
                    samples=INTERLACE,
                    window='cic'):

        comm = source.comm
        # source here must be a CatalogSource
        assert isinstance(source, CatalogSourceBase)

        if BoxSize is None:
            BoxSize = source.attrs['BoxSize']

        # copy meta-data from source too
        self.attrs.update(source.attrs)

        # copy over the necessary meta-data to attrs
        self.attrs['BoxSize'] = BoxSize
        self.attrs['Nmesh'] = Nmesh
        self.attrs['window'] = window

        # store others as straight attributes
        self.dtype = dtype
        self.weight = weight
        self.value = value
        self.selection = selection
        self.position = position
        self.samples = samples

        # add in the Mesh Source attributes
        MeshSource.__init__(self, comm, Nmesh, BoxSize, dtype)

        # finally set the base as the input CatalogSource
        # NOTE: set this AFTER MeshSource.__init__()
        self.base = source

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
        self.attrs['window'] = value.lower() # lower to compare with compensation

    def to_complex_field(self, out=None, normalize=True):
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
        complex: :class:`pmesh.pm.RealField`
            the painted real field; this has a ``attrs`` dict storing meta-data
        """
        # check for 'Position' column
        if self.position not in self.base:
            msg = "in order to paint a CatalogSource to a RealField, add a "
            msg += "column named '%s', representing the particle positions" %self.position
            raise ValueError(msg)

        pm = self.pm
        Nlocal = 0 # (unweighted) number of particles read on local rank
        Wlocal = 0 # (weighted) number of particles read on local rank

        # the paint brush window
        paintbrush = window.methods[self.window]

        # initialize the ComplexField to return
        if out is not None:
            assert isinstance(out, ComplexField), "output of to_complex_field must be a ComplexField"
            numpy.testing.assert_array_equal(out.pm.Nmesh, pm.Nmesh)
            toret = out
        else:
            toret = ComplexField(pm)
        toret[:] = 0

        # displaced aa fields
        reals = [pm.create(mode='real', value=0) for vector in self.samples]

        # read the necessary data (as dask arrays)
        columns = [self.position, self.weight, self.value, self.selection]

        Position, Weight, Value, Selection = self.base.read(columns)

        # ensure the slices are synced, since decomposition is collective
        Nlocalmax = max(pm.comm.allgather(len(Position)))

        # paint data in chunks on each rank;
        # we do this by chunk 8 million is pretty big anyways.
        chunksize = _global_options['paint_chunk_size']
        for i in range(0, Nlocalmax, chunksize):
            s = slice(i, i + chunksize)

            if len(Position) != 0:

                # selection has to be computed many times when data is `large`.
                sel = self.base.compute(Selection[s])

                # be sure to use the source to compute
                position, weight, value = \
                    self.base.compute(Position[s], Weight[s], Value[s])

                # FIXME: investigate if move selection before compute
                # speeds up IO.
                position = position[sel]
                weight = weight[sel]
                value = value[sel]
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

            lay = pm.decompose(position, smoothing=max([1.0 * paintbrush.support, 1.0]))
            p = lay.exchange(position)
            w = lay.exchange(weight)
            v = lay.exchange(value)

            H = pm.BoxSize / pm.Nmesh

            for real, vector in zip(reals, self.samples):
                # in mesh units
                shifted = pm.affine.shift(vector)

                pm.paint(p, mass=w * v, resampler=paintbrush, transform=shifted, hold=True, out=real)

            Nglobal = pm.comm.allreduce(Nlocal)

            if pm.comm.rank == 0:
                self.logger.info("painted %d out of %d objects to mesh"
                    % (Nglobal, self.base.csize))

        # now the loop over particles is done

        # add the aa fields:
        for real, vector in zip(reals, self.samples):
            # compose the two interlaced fields into the final result.

            def filter(k, v, vector=vector):
                kH = sum(ki * Hi for ki, Hi in zip(k, vector * pm.BoxSize / pm.Nmesh))
                return numpy.exp(1j * kH) * v

            c = real.r2c(out=Ellipsis).apply(filter, out=Ellipsis)
            toret[...] += c

        # it's the average of all aa fields.
        toret /= len(self.samples)

        toret.apply(self._get_compensation(), out=Ellipsis, kind='circular')

        # unweighted number of objects
        N = pm.comm.allreduce(Nlocal)

        # weighted number of objects
        W = pm.comm.allreduce(Wlocal)

        # weighted number density (objs/cell)
        nbar = 1. * W / numpy.prod(pm.Nmesh)

        # make sure we painted something or nbar is nan; in which case
        # we set the density to uniform everywhere.
        if N == 0:
            warnings.warn(("trying to paint particle source to mesh, "
                           "but no particles were found!"),
                            RuntimeWarning
                        )

        # shot noise is volume / un-weighted number
        shotnoise = numpy.prod(pm.BoxSize) / N

        # save some meta-data
        toret.attrs = {}
        toret.attrs['shotnoise'] = shotnoise
        toret.attrs['N'] = N
        toret.attrs['W'] = W
        toret.attrs['num_per_cell'] = nbar

        if pm.comm.rank == 0:
            self.logger.info("painted %d out of %d objects to mesh" %(N,self.base.csize))
            self.logger.info("mean particles per cell is %g", nbar)
            self.logger.info("normalized the convention to 1 + delta")

        if normalize:
            if nbar > 0:
                toret[...] /= nbar
            else:
                toret[...] = 1

        return toret

    def _get_compensation(self):
        """
        Return the compensation function, which corrects for the
        windowing kernel.

        The compensation function is computed as:

        - if ``interlaced = True``:
          - :func:`CompensateCIC` if using CIC window
          - :func:`CompensateTSC` if using TSC window
          - :func:`CompensatePCS` if using PCS window
        """
        d = {'cic' : self.CompensateCIC,
             'tsc' : self.CompensateTSC,
             'pcs' : self.CompensatePCS,
            }

        if not self.window in d:
            raise ValueError("compensation for window %s is not defined" % self.window)

        return d[self.window]

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
    def CompensatePCS(w, v):
        """
        Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the PCS window function in configuration space.

        .. note::
            see equation 18 (with p=4) of
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
            tmp = (numpy.sinc(0.5 * wi / numpy.pi) ) ** 4
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
