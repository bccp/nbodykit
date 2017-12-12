from nbodykit.base.mesh import MeshSource
from nbodykit.base.catalog import CatalogSource, CatalogSourceBase
from nbodykit import _global_options
import numpy
import logging
import warnings

# for converting from particle to mesh
from pmesh import window
from pmesh.pm import RealField, ComplexField

class CatalogMesh(CatalogSource, MeshSource):
    """
    A view of a CatalogSource object which knows how to create a MeshSource
    object from itself.

    The original CatalogSource object is stored as the :attr:`base` attribute.

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
    interlaced : bool, optional
        use the interlacing technique of Sefusatti et al. 2015 to reduce
        the effects of aliasing on Fourier space quantities computed
        from the mesh
    compensated : bool, optional
        whether to correct for the window introduced by the grid
        interpolation scheme
    window : str, optional
        the string specifying which window interpolation scheme to use;
        see ``pmesh.window.methods``
    """
    logger = logging.getLogger('CatalogMesh')

    def __repr__(self):
        if isinstance(self.base, CatalogMesh):
            return repr(self.base)
        else:
            return "(%s as CatalogMesh)" % repr(self.base)

    def __new__(cls, source, BoxSize, Nmesh, dtype, weight,
                    value, selection, position='Position', interlaced=False,
                    compensated=False, window='cic', **kwargs):

        # source here must be a CatalogSource
        assert isinstance(source, CatalogSourceBase)

        # new, empty CatalogSource
        obj = CatalogSourceBase.__new__(cls, source.comm)

        # copy over size from the CatalogSource
        obj._size = source.size
        obj._csize = source.csize

        # copy over the necessary meta-data to attrs
        obj.attrs['BoxSize'] = BoxSize
        obj.attrs['Nmesh'] = Nmesh
        obj.attrs['interlaced'] = interlaced
        obj.attrs['compensated'] = compensated
        obj.attrs['window'] = window

        # copy meta-data from source too
        obj.attrs.update(source.attrs)

        # store others as straight attributes
        obj.dtype = dtype
        obj.weight = weight
        obj.value = value
        obj.selection = selection
        obj.position = position

        # add in the Mesh Source attributes
        MeshSource.__init__(obj, obj.comm, Nmesh, BoxSize, dtype)

        # finally set the base as the input CatalogSource
        # NOTE: set this AFTER MeshSource.__init__()
        obj.base = source

        return obj

    def gslice(self, start, stop, end=1, redistribute=True):
        """
        Execute a global slice of a CatalogMesh.

        .. note::
            After the global slice is performed, the data is scattered
            evenly across all ranks.

        As CatalogMesh objects are views of a CatalogSource, this simply
        globally slices the underlying CatalogSource.

        Parameters
        ----------
        start : int
            the start index of the global slice
        stop : int
            the stop index of the global slice
        step : int, optional
            the default step size of the global size
        redistribute : bool, optional
            if ``True``, evenly re-distribute the sliced data across all
            ranks, otherwise just return any local data part of the global
            slice
        """
        # sort the base object
        newbase = self.base.gslice(start, stop, end=end, redistribute=redistribute)

        # view this base class as a CatalogMesh (with default CatalogMesh parameters)
        toret = newbase.view(self.__class__)

        # attach the meta-data from self to returned sliced CatalogMesh
        return toret.__finalize__(self)

    def sort(self, keys, reverse=False, usecols=None):
        """
        Sort the CatalogMesh object globally across all MPI ranks
        in ascending order by the input keys.

        Sort columns must be floating or integer type.

        As CatalogMesh objects are views of a CatalogSource, this simply
        sorts the underlying CatalogSource.

        Parameters
        ----------
        *keys :
            the names of columns to sort by. If multiple columns are provided,
            the data is sorted consecutively in the order provided
        reverse : bool, optional
            if ``True``, perform descending sort operations
        usecols : list, optional
            the name of the columns to include in the returned CatalogSource
        """
        # sort the base object
        newbase = self.base.sort(keys, reverse=reverse, usecols=usecols)

        # view this base class as a CatalogMesh (with default CatalogMesh parameters)
        toret = newbase.view(self.__class__)

        # attach the meta-data from self to returned sliced CatalogMesh
        return toret.__finalize__(self)

    def __slice__(self, index):
        """
        Return a slice of a CatalogMesh object.

        This slices the CatalogSource object stored as the :attr:`base`
        attribute, and then views that sliced object as a CatalogMesh.

        Parameters
        ----------
        index : array_like
            either a dask or numpy boolean array; this determines which
            rows are included in the returned object

        Returns
        -------
        subset
            the particle source with the same meta-data as ``self``, and
            with the sliced data arrays
        """
        # this slice of the CatalogSource will be the base of the mesh
        base = super(CatalogMesh, self).__slice__(index)

        # view this base class as a CatalogMesh (with default CatalogMesh parameters)
        toret = base.view(self.__class__)

        # attach the meta-data from self to returned sliced CatalogMesh
        return toret.__finalize__(self)

    def copy(self):
        """
        Return a shallow copy of ``self``.

        .. note::
            No copy of data is made.

        Returns
        -------
        CatalogMesh :
            a new CatalogMesh that holds all of the data columns of ``self``
        """
        # copy the base and view it as a CatalogMesh
        toret = self.base.copy().view(self.__class__)

        # attach the meta-data from self to returned sliced CatalogMesh
        return toret.__finalize__(self)

    def __finalize__(self, other):
        """
        Finalize the creation of a CatalogMesh object by copying over
        attributes from a second CatalogMesh.

        This also copies over the relevant MeshSource attributes via a
        call to :func:`MeshSource.__finalize__`.

        Parameters
        ----------
        obj : CatalogMesh
            the second CatalogMesh to copy over attributes from
        """
        if isinstance(other, CatalogSourceBase):
            self = CatalogSourceBase.__finalize__(self, other)

        if isinstance(other, MeshSource):
            self = MeshSource.__finalize__(self, other)

        return self

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
        self.attrs['window'] = value.lower() # lower to compare with compensation

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

        # initialize the RealField to return
        if out is not None:
            assert isinstance(out, RealField), "output of to_real_field must be a RealField"
            numpy.testing.assert_array_equal(out.pm.Nmesh, pm.Nmesh)
            toret = out
        else:
            toret = RealField(pm)
            toret[:] = 0

        # for interlacing, we need two empty meshes if out was provided
        # since out may have non-zero elements, messing up our interlacing sum
        if self.interlaced:

            real1 = RealField(pm)
            real1[:] = 0

            # the second, shifted mesh (always needed)
            real2 = RealField(pm)
            real2[:] = 0

        # read the necessary data (as dask arrays)
        columns = [self.position, self.weight, self.value, self.selection]

        Position, Weight, Value, Selection = self.read(columns)

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

            # no interlacing
            if not self.interlaced:
                lay = pm.decompose(position, smoothing=0.5 * paintbrush.support)
                p = lay.exchange(position)
                w = lay.exchange(weight)
                v = lay.exchange(value)
                pm.paint(p, mass=w * v, resampler=paintbrush, hold=True, out=toret)

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
                pm.paint(p, mass=w * v, resampler=paintbrush, hold=True, out=real1)
                pm.paint(p, mass=w * v, resampler=paintbrush, transform=shifted, hold=True, out=real2)

            Nglobal = pm.comm.allreduce(Nlocal)

            if pm.comm.rank == 0:
                self.logger.info("painted %d out of %d objects to mesh"
                    % (Nglobal, self.base.csize))

        # now the loop over particles is done

        if not self.interlaced:
            # nothing to do, toret is already filled.
            pass
        else:
            # compose the two interlaced fields into the final result.
            c1 = real1.r2c()
            c2 = real2.r2c()

            # and then combine
            for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                kH = sum(k[i] * H[i] for i in range(3))
                s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * numpy.exp(0.5 * 1j * kH)

            # FFT back to real-space
            # NOTE: cannot use "toret" here in case user supplied "out"
            c1.c2r(real1)

            # need to add to the returned mesh if user supplied "out"
            toret[:] += real1[:]


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

        csum = toret.csum()
        if pm.comm.rank == 0:
            self.logger.info("painted %d out of %d objects to mesh" %(N,self.base.csize))
            self.logger.info("mean particles per cell is %g", nbar)
            self.logger.info("sum is %g ", csum)
            self.logger.info("normalized the convention to 1 + delta")

        if normalize:
            if nbar > 0:
                toret[...] /= nbar
            else:
                toret[...] = 1

        return toret

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
          - :func:`CompensatePCS` if using PCS window
        - if ``interlaced = False``:
          - :func:`CompensateCICShotnoise` if using CIC window
          - :func:`CompensateTSCShotnoise` if using TSC window
          - :func:`CompensatePCSShotnoise` if using PCS window
        """
        if self.interlaced:
            d = {'cic' : self.CompensateCIC,
                 'tsc' : self.CompensateTSC,
                 'pcs' : self.CompensatePCS,
                }
        else:
            d = {'cic' : self.CompensateCICShotnoise,
                 'tsc' : self.CompensateTSCShotnoise,
                 'pcs' : self.CompensatePCSShotnoise,
                }

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

    @staticmethod
    def CompensateTSCShotnoise(w, v):
        """
        Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the TSC window function in configuration space,
        as well as the approximate aliasing correction to the first order

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
    def CompensatePCSShotnoise(w, v):
        """
        Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the PCS window function in configuration space,
        as well as the approximate aliasing correction to first order

        .. note::

            YF: I derived this by fitting the result to s ** 3
            according to the form given in equation 20 of Jing et al.
            It must be possible to derive this manually.

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
            v = v / (1 - 4./3. * s + 2./5. * s**2 - 4./315. * s**3) ** 0.5
        return v

    @staticmethod
    def CompensateCICShotnoise(w, v):
        """
        Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the CIC window function in configuration space,
        as well as the approximate aliasing correction to the first order

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
