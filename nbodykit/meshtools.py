import numpy

class MeshSlab(object):
    """
    A convenience class to represent a specific slab of a mesh,
    which is denoted as a ``slab``
    """
    def __init__(self, islab, coords, axis, symmetry_axis):
        """
        Parameters
        ----------
        islab : int, [0, x[0].shape[0]]
            the index of the slab, which indexes the first dimension of
            the mesh (the `x` coordinate), thus producing a y-z plane
        coords : list of arrays
            the coordinate arrays of the mesh, with proper 3D shapes
            for easy broadcasting; if the mesh has size (Nx, Ny, Nz),
            then the shapes of `x` are: [(Nx, 1, 1), (1, Ny, 1), (1, 1, Nz)]
        axis : int, {0, 1, 2}
            the index of the mesh axis to iterate over
        symmetry_axis : int, optional
            if provided, the axis that has been compressed due to Hermitian symmetry
        """
        self.ndim          = len(coords)
        self._index        = islab
        self.axis          = axis
        self.symmetry_axis = symmetry_axis
        self._coords       = coords

        # make sure symmetry_axis > 0 (sanity check)
        if self.hermitian_symmetric and self.symmetry_axis < 0:
            raise ValueError("`symmetry_axis` in MeshSlab must be non-negative")

    def __str__(self):
        name = self.__class__.__name__
        return "<%s: axis=%d, index=%d>" %(name, self.axis, self._index)

    def __repr__(self):
        return self.__str__()

    @property
    def index(self):
        """
        Return an indexing list appropriate for selecting the slicing the
        appropriate slab out of a full 3D array of shape (Nx, Ny, Nz)
        """
        toret = [slice(None)]*self.ndim
        toret[self.axis] = self._index
        return tuple(toret)

    @property
    def meshshape(self):
        """
        Return the local shape of the mesh on this rank, as determined
        by the input coordinates array
        """
        return tuple([numpy.shape(self._coords[i])[i] for i in range(self.ndim)])

    @property
    def shape(self):
        """
        Return the shape of the slab
        """
        return tuple([s for i, s in enumerate(self.meshshape) if i != self.axis])

    @property
    def hermitian_symmetric(self):
        """
        Whether the slab is Hermitian-symmetric
        """
        return self.symmetry_axis is not None

    def coords(self, i):
        """
        Return the coordinate array for dimension ``i`` on this slab,

        .. note::

            The return value will be properly squeezed for easy
            broadcasting, i.e., if `i` is `self.axis`, then an array of
            shape `(1,1)` array is returned, otherwise, the shape is
            `(N_i, 1)` or `(1, N_i)`

        Parameters
        ----------
        i : int, {0,1,2}
            the index of the desired dimension

        Returns
        -------
        array_like
            the coordinate array for dimension `i` on the slab; see the
            note about the shape of the return array for details
        """
        if i < 0: i += self.ndim
        assert 0 <= i < self.ndim, "i should be between 0 and %d" %self.ndim

        if i != self.axis:
            return numpy.take(self._coords[i], 0, axis=self.axis)
        else:
            return numpy.take(self._coords[i], self._index, axis=self.axis)

    def norm2(self):
        """
        The square of coordinate grid norm defined at each
        point on the slab.

        This broadcasts the coordinate arrays along each dimension
        to compute the norm at each point in the slab.

        Returns
        -------
        array_like, (slab.shape)
            the square of coordinate mesh at each point in the slab
        """
        return sum(self.coords(i)**2 for i in range(self.ndim))

    def mu(self, los):
        """
        The `mu` value defined at each point on the slab for the
        specified line-of-sight index


        Parameters
        ----------
        los: array_like,
            the direction of the line-of-sight, which `mu` is defined
            with respect to; must have a norm of 1.

        Returns
        -------
        array_like, (slab.shape)
            the `mu` value at each point in the slab
        """
        with numpy.errstate(invalid='ignore', divide='ignore'):
            return sum(self.coords(i) * los[i] for i in range(self.ndim)) / self.norm2()**0.5

    @property
    def nonsingular(self):
        """
        The indices on the slab of the positive frequencies
        along the dimension specified by `symmetry_axis`.

        This takes advantage of the fact that :class:`pmesh.pm.ComplexField`
        shifts the Nyquist frequencies to the negative halves.
        Therefore this ensures that the zero and Nyquist planes perpendicular
        to the symmetry axis have weight 1, whereas other modes have weight 2.

        Returns
        -------
        idx : array_like, self.shape
            Return a boolean array with the shape of the slab,
            with `True` elements giving the elements with
            positive freq along `symmetry_axis`
        """
        try:
            return self._nonsingular
        except AttributeError:

            # initially, return slice that includes all elements
            idx = numpy.ones(self.shape, dtype=bool)

            # iteration axis is symmetry axis
            if self.symmetry_axis == self.axis:

                # check if current iteration value is positive
                if numpy.float(self.coords(self.axis)) <= 0.:
                    idx = numpy.zeros(self.shape, dtype=bool)

            # one of slab dimensions is symmetry axis
            else:

                # get the indices that have positive freq along symmetry axis
                nonsingular = (self._coords[self.symmetry_axis] > 0.)
                nonsingular = numpy.take(nonsingular, 0, axis=self.axis)

                idx[...] = nonsingular

            self._nonsingular = idx
            return self._nonsingular

    @property
    def hermitian_weights(self):
        """
        Weights to be applied to quantities on the slab in order
        to account for Hermitian symmetry

        These weights double-count the positive frequencies along
        the `symmetry_axis`.
        """
        try:
            return self._weights
        except AttributeError:

            # if not Hermitian symmetric, weights are 1
            if not self.hermitian_symmetric:
                toret = 1.
            # iteration axis is symmetry axis
            elif self.axis == self.symmetry_axis:
                toret = 1.
                if numpy.float(self.coords(self.symmetry_axis)) > 0.:
                    toret = 2.
            # only nonsingular plane gets factor of 2
            else:
                toret = numpy.ones(self.shape, dtype='f4')
                toret[self.nonsingular] = 2.

            self._weights = toret
            return self._weights

def SlabIterator(coords, axis=0, symmetry_axis=None):
    """
    Iterate over the specified dimension of the coordinate mesh,
    returning a :class:`MeshSlab` for each iteration

    Parameters
    ----------
    coords : list of arrays
        the coordinate arrays of the mesh, with proper 3D shapes
        for easy broadcasting; if the mesh has size (Nx, Ny, Nz),
        then the shapes of `x3d` should be: ``[(Nx, 1, 1), (1, Ny, 1), (1, 1, Nz)]``
    axis : int, optional
        the index of the mesh axis to iterate over
    symmetry_axis : int, optional
        if provided, the axis that has been compressed due to Hermitian symmetry
    """
    # number of dimensions in the mesh
    ndim = len(coords)
    if ndim not in [2,3]:
        raise NotImplementedError("SlabIterator can only be used on 3D or 2D arrays")

    # account for negative axes
    if axis < 0: axis += ndim
    assert 0 <= axis < ndim, "axis should be between 0 and %d" %ndim
    if symmetry_axis is not None and symmetry_axis < 0: symmetry_axis += ndim

    # this will only work if shapes are: [(Nx, 1, 1), (1, Ny, 1), (1, 1, Nz)]
    # mainly a sanity check to make sure things work
    shapes = [numpy.shape(x) for x in coords]
    try:
        mesh_size = [shape[i] for i, shape in enumerate(shapes)]
        for i in range(ndim):
            desired_shape = numpy.ones(ndim)
            desired_shape[i] = mesh_size[i]
            if shapes[i] != tuple(desired_shape):
                raise ValueError("coordinate array shape mismatch")
    except:
        raise ValueError("input coordinates with shapes %s are not correct" %str(shapes))

    # iterate over the specified axis, slab by slab
    N = numpy.shape(coords[axis])[axis]
    for islab in range(N):
        yield MeshSlab(islab, coords, axis, symmetry_axis)
