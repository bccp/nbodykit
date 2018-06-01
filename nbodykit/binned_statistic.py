import numpy

def bin_ndarray(ndarray, new_shape, weights=None, operation=numpy.mean):
    """
    Bins an ndarray in all axes based on the target shape, by summing
    or averaging.

    Parameters
    ----------
    ndarray : array_like
        the input array to re-bin
    new_shape : tuple
        the tuple holding the desired new shape
    weights : array_like, optional
        weights to multiply the input array by, before running the re-binning
        operation,

    Notes
    -----
    *   Dimensions in `new_shape` must be integral factor smaller than the
        old shape
    *   Number of output dimensions must match number of input dimensions.
    *   See https://gist.github.com/derricw/95eab740e1b08b78c03f

    Examples
    --------
    >>> m = numpy.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation=numpy.sum)
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if ndarray.shape == new_shape:
        raise ValueError("why are we re-binning if the new shape equals the old shape?")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    if numpy.sometrue(numpy.mod(ndarray.shape, new_shape)):
        args = (str(new_shape), str(ndarray.shape))
        msg = "desired shape of %s must be integer factor smaller than the old shape %s" %args
        raise ValueError(msg)

    compression_pairs = [(d, c//d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    if weights is not None: weights = weights.reshape(flattened)

    for i in range(len(new_shape)):
        if weights is not None:
            ndarray = operation(ndarray*weights, axis=-1*(i+1))
            weights = numpy.sum(weights, axis=-1*(i+1))
            ndarray /= weights
        else:
            ndarray = operation(ndarray, axis=-1*(i+1))
    return ndarray

class BinnedStatistic(object):
    """
    Lightweight class to hold statistics binned at fixed coordinates.

    For example, this class could hold a grid of (r, mu) or (k, mu) bins
    for a correlation function or power spectrum measurement.

    It is modeled after the syntax of :class:`xarray.Dataset`, and is designed
    to hold correlation function or power spectrum results (in 1D or 2D)

    Parameters
    ----------
    dims : list, (Ndim,)
        A list of strings specifying names for the coordinate dimensions.
        The dimension names stored in :attr:`dims` have the suffix 'cen'
        added, to indicate that the coordinate grid is defined at the bin
        centers
    edges : list, (Ndim,)
        A list specifying the bin edges for each dimension
    data : array_like
        a structured array holding the data variables, where the named
        fields interpreted as the variable names. The variable names are
        stored in :attr:`variables`
    fields_to_sum : list, optional
        the name of fields that will be summed when reindexing, instead
        of averaging
    **kwargs :
        Any additional keywords are saved as metadata in the :attr:`attrs`
        dictionary attribute

    Examples
    --------
    The following example shows how to read a power spectrum
    measurement from a JSON file, as output by nbodykit, assuming
    the JSON file holds a dictionary with a 'power' entry holding the
    relevant data

    >>> filename = 'test_data.json'
    >>> pk = BinnedStatistic.from_json(['k'], filename, 'power')

    In older versions of nbodykit, results were written using plaintext ASCII
    files. Although now deprecated, this type of files can be read using:

    >>> filename = 'test_data.dat'
    >>> dset = BinnedStatistic.from_plaintext(['k'], filename)

    Data variables can be accessed in a dict-like fashion:

    >>> power = pkmu['power'] # returns power data variable

    Array-like indexing of a :class:`BinnedStatistic` returns a new :class:`BinnedStatistic`
    holding the sliced data:

    >>> pkmu
    <BinnedStatistic: dims: (k: 200, mu: 5), variables: ('mu', 'k', 'power')>
    >>> pkmu[:,0] # select first mu column
    <BinnedStatistic: dims: (k: 200), variables: ('mu', 'k', 'power')>

    Additional data variables can be added to the :class:`BinnedStatistic` via:

    >>> modes = numpy.ones((200, 5))
    >>> pkmu['modes'] = modes

    Coordinate-based indexing is possible through :func:`sel`:

    >>> pkmu
    <BinnedStatistic: dims: (k: 200, mu: 5), variables: ('mu', 'k', 'power')>
    >>> pkmu.sel(k=slice(0.1, 0.4), mu=0.5)
    <BinnedStatistic: dims: (k: 30), variables: ('mu', 'k', 'power')>

    :func:`squeeze` will explicitly squeeze the specified dimension
    (of length one) such that the resulting instance has one less dimension:

    >>> pkmu
    <BinnedStatistic: dims: (k: 200, mu: 1), variables: ('mu', 'k', 'power')>
    >>> pkmu.squeeze(dim='mu') # can also just call pkmu.squeeze()
    <BinnedStatistic: dims: (k: 200), variables: ('mu', 'k', 'power')>

    :func:`average` returns a new :class:`BinnedStatistic` holding the
    data averaged over one dimension

    :func:`reindex` will re-bin the coordinate arrays along the specified
    dimension
    """
    def __init__(self, dims, edges, data, fields_to_sum=[], coords=None, **kwargs):

        # number of dimensions must match
        if len(dims) != len(edges):
            raise ValueError("size mismatch between specified `dims` and `edges`")

        # input data must be structured array
        if not isinstance(data, numpy.ndarray) or data.dtype.names is None:
            raise TypeError("'data' should be a structured numpy array")

        shape = tuple(len(e)-1 for e in edges)
        if data.shape != shape:
            args = (shape, data.shape)
            raise ValueError("`edges` imply data shape of %s, but data has shape %s" %args)

        self.dims = list(dims)
        self.edges = dict(zip(self.dims, edges))

        # coordinates are the bin centers
        self.coords = {}
        for i, dim in enumerate(self.dims):
            if coords is not None and coords[i] is not None:
                self.coords[dim] = numpy.copy(coords[i])
            else:
                self.coords[dim] = 0.5 * (edges[i][1:] + edges[i][:-1])

        # store variables as a structured array
        self.data = data.copy()

        # define a mask such that a coordinate grid element will be masked
        # if any of the variables at that coordinate are (NaN, inf)
        self.mask = numpy.zeros(self.shape, dtype=bool)
        for name in data.dtype.names:
            self.mask = numpy.logical_or(self.mask, ~numpy.isfinite(self.data[name]))

        # fields that we wish to sum, instead of averaging
        self._fields_to_sum = fields_to_sum

        # save and track metadata
        self.attrs = {}
        for k in kwargs: self.attrs[k] = kwargs[k]

    @classmethod
    def from_state(kls, state):
        obj = kls(dims=state['dims'], edges=state['edges'], coords=state['coords'], data=state['data'])
        obj.attrs.update(state['attrs'])
        return obj

    def __getstate__(self):
        return dict(
                dims=self.dims,
                edges=[self.edges[dim] for dim in self.dims],
                coords=[self.coords[dim] for dim in self.dims],
                data=self.data,
                attrs=self.attrs)

    @property
    def shape(self):
        """
        The shape of the coordinate grid
        """
        return tuple(len(self.coords[d]) for d in self.dims)

    @property
    def variables(self):
        """
        Alias to return the names of the variables stored in `data`
        """
        return list(self.data.dtype.names)

    @classmethod
    def __construct_direct__(cls, data, mask, **kwargs):
        """
        Shortcut around __init__ for internal use to construct and
        return a new class instance. The returned object should be
        identical to that returned by __init__.

        Notes
        -----
        *   Useful for returning new instances with sliced data/mask
        *   The keyword arguments required to create a full, unbroken
            instance are `dims`, `coords`, `edges`, and `attrs`

        Parameters
        ----------
        data :
        """
        obj = object.__new__(cls)
        for k in kwargs: setattr(obj, k, kwargs[k])

        for k, d in zip(['data', 'mask'], [data, mask]):
            setattr(obj, k, d)
            if obj.shape != d.shape:
                try:
                    setattr(obj, k, d.reshape(obj.shape))
                except:
                    raise ValueError("shape mismatch between data and coordinates")
        return obj

    def __copy_attrs__(self):
        """
        Return a copy of all necessary attributes associated with
        the `BinnedStatistic`. This dictionary + `data` and `mask` are all
        that's required to reconstruct a new class
        """
        kw = {}
        kw['dims'] = list(self.dims)
        kw['edges'] = self.edges.copy()
        kw['coords'] = self.coords.copy()
        kw['attrs'] = self.attrs.copy()
        kw['_fields_to_sum'] = list(self._fields_to_sum)
        return kw

    def __finalize__(self, data, mask, indices):
        """
        Finalize and return a new instance from a slice of the
        current object (returns a copy)
        """
        edges, coords = self.__slice_edges__(indices)
        kw = {'dims':list(self.dims), 'edges':edges, 'coords':coords,
              'attrs':self.attrs.copy(), '_fields_to_sum':self._fields_to_sum}
        return self.__class__.__construct_direct__(data, mask, **kw)

    def __slice_edges__(self, indices):
        """
        Internal function to slice the `edges` attribute with the
        specified indices, which specify the included coordinate bins
        """
        edges = {}
        coords = {}
        for i, dim in enumerate(self.dims):
            if len(indices[i]) > 0:
                idx = list(indices[i]) + [indices[i][-1]+1]
            else:
                idx = [0]
            edges[dim] = self.edges[dim][idx]
            coords[dim] = 0.5 * (edges[dim][1:] + edges[dim][:-1])

        return edges, coords

    def __str__(self):
        name = self.__class__.__name__
        dims = "(" + ", ".join(['%s: %d' %(k, self.shape[i]) for i, k in enumerate(self.dims)]) + ")"

        if len(self.variables) < 5:
            return "<%s: dims: %s, variables: %s>" %(name, dims, str(tuple(self.variables)))
        else:
            return "<%s: dims: %s, variables: %d total>" %(name, dims, len(self.variables))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.variables)

    def __contains__(self, key):
        return key in self.variables

    def __setitem__(self, key, data):
        """
        Add a new variable with the name `key` to the class using `data`
        """
        if numpy.shape(data) != self.data.shape:
            raise ValueError("data to be added must have shape %s" %str(self.data.shape))

        # add the new (key, type) to the dtype, or if the key is present, overwrite
        dtype = list(self.data.dtype.descr) # make copy
        names = list(self.data.dtype.names) # make copy
        if key in names:
            i = names.index(key)
            dtype.pop(i); names.pop(i)
        dtype += [(key, data.dtype.type)]

        # add old variables
        dtype = numpy.dtype(dtype)
        new = numpy.zeros(self.data.shape, dtype=dtype)
        for col in names:
            new[col] = self.data[col]

        # the new data to add
        new[key] = data
        mask = numpy.logical_or(self.mask, ~numpy.isfinite(new[key]))

        # save
        self.data = new
        self.mask = mask

    def __getitem__(self, key):
        """
        Index- or string- based indexing

        Notes
        -----
        *   If a single string is passed, the key is intrepreted
            as a `variable` or `coordinate`, and the corresponding
            array is returned
        *   If a list of strings is passed, then a new `BinnedStatistic`
            holding only the `variable` names in `key` is returned
        *   Integer-based indexing or slices similar to numpy
            indexing will slice `data`, returning a new
            `BinnedStatistic` holding the newly sliced data and coordinate grid
        *   Scalar indexes (i.e., integers) used to index a certain
            dimension will "squeeze" that dimension, removing it
            from the coordinate grid
        """
        # if single string passed, return a coordinate or variable
        if isinstance(key, str):
            if key in self.variables:
                return self.data[key]
            else:
                raise KeyError("`%s` is not a valid variable name" %key)

        # indices to slice the data with
        indices = [list(range(0, self.shape[i])) for i in range(len(self.dims))]

        # check for list/tuple of variable names
        # if so, return a BinnedStatistic with slice of columns
        if isinstance(key, (list, tuple)) and all(isinstance(x, str) for x in key):
            if all(k in self.variables for k in key):
                return self.__finalize__(self.data[list(key)], self.mask.copy(), indices)
            else:
                invalid = ', '.join("'%s'" %k for k in key if k not in self.variables)
                raise KeyError("cannot slice variables -- invalid names: (%s)" %invalid)

        key_ = key

        # if key is a single integer or single slice, make it a list
        make_iterable = isinstance(key, slice) or isinstance(key, int)
        if make_iterable or isinstance(key, list) and all(isinstance(x, int) for x in key):
            key_ = [key]

        squeezed_dims = []
        for i, subkey in enumerate(key_):
            if i >= len(self.dims):
                raise IndexError("too many indices for BinnedStatistic; note that ndim = %d" %len(self.dims))

            if isinstance(subkey, int):
                indices[i] = [subkey]
                squeezed_dims.append(self.dims[i])
            elif isinstance(subkey, list):
                indices[i] = subkey
            elif isinstance(subkey, slice):
                indices[i] = list(range(*subkey.indices(self.shape[i])))

        # can't squeeze all dimensions!!
        if len(squeezed_dims) == len(self.dims):
            raise IndexError("cannot return object with all remaining dimensions squeezed")

        # fail nicely if we fail at all
        try:
            toret = self.__finalize__(self.data[key], self.mask[key], indices)
            for dim in squeezed_dims:
                toret = toret.squeeze(dim)
            return toret
        except ValueError:
            raise IndexError("this type of slicing not implemented")

    def _get_index(self, dim, val, method=None):
        """
        Internal function to compute the bin index of the nearest
        coordinate value to the input value
        """
        index = self.coords[dim]
        if method == 'nearest':
            i = (numpy.abs(index-val)).argmin()
        else:
            try:
                i = list(index).index(val)
            except Exception as e:
                args = (dim, str(e))
                msg = "error converting '%s' index; try setting `method = 'nearest'`: %s"
                raise IndexError(msg %args)

        return i

    #--------------------------------------------------------------------------
    # user-called functions
    #--------------------------------------------------------------------------
    def to_json(self, filename):
        """
        Write a BinnedStatistic from a JSON file.

        .. note::
            This uses :class:`nbodykit.utils.JSONEncoder` to write the
            JSON file

        Parameters
        ----------
        filename : str
            the name of the file to write
        """
        import json
        from nbodykit.utils import JSONEncoder
        state = self.__getstate__()
        with open(filename, 'w') as ff:
            json.dump(state, ff, cls=JSONEncoder)

    @classmethod
    def from_json(cls, filename, key='data', dims=None, edges=None, **kwargs):
        """
        Initialize a BinnedStatistic from a JSON file.

        The JSON file should contain a dictionary, where the data to load is stored
        as the ``key`` entry, with an ``edges`` entry specifying bin edges, and
        optionally, a ``attrs`` entry giving a dict of meta-data

        .. note::
            This uses :class:`nbodykit.utils.JSONDecoder` to load the
            JSON file

        Parameters
        ----------
        filename : str
            the name of the file to load
        key : str, optional
            the name of the key in the JSON file holding the data to load
        dims : list, optional
            list of names specifying the dimensions, i.e., ``['k']`` or ``['k', 'mu']``;
            must be supplied if not given in the JSON file

        Returns
        -------
        dset : BinnedStatistic
            the BinnedStatistic holding the data from file
        """
        import json
        from nbodykit.utils import JSONDecoder

        # parse
        with open(filename, 'r') as ff:
            state = json.load(ff, cls=JSONDecoder)

        # the data
        if key not in state:
            args = (key, tuple(state.keys()))
            raise ValueError("no data entry found in JSON format for '%s' key; valid keys are %s" %args)
        data = state[key]

        # the dimensions
        dims = state.pop('dims', dims)
        if dims is None:
            raise ValueError("no `dims` found in JSON file; please specify as keyword argument")

        # the edges
        edges = state.pop('edges', edges)
        if edges is None:
            raise ValueError("no `edges` found in JSON file; please specify as keyword argument")

        # the coords
        coords = state.pop('coords', None)

        # meta-data
        attrs = state.pop('attrs', {})
        attrs.update(kwargs)

        return cls(dims, edges, data, coords=coords, **attrs)

    @classmethod
    def from_plaintext(cls, dims, filename, **kwargs):
        """
        Initialize a BinnedStatistic from a plaintext file

        .. note:: Deprecated in nbodykit 0.2.x
            Storage of BinnedStatistic objects as plaintext ASCII files is no longer supported;
            See :func:`BinnedStatistic.from_json`

        Parameters
        ----------
        dims : list
            list of names specifying the dimensions, i.e., ``['k']`` or ``['k', 'mu']``
        filename : str
            the name of the file to load

        Returns
        -------
        dset : BinnedStatistic
            the BinnedStatistic holding the data from file
        """
        import warnings
        msg = "storage of BinnedStatistic objects as ASCII plaintext files is deprecated; see BinnedStatistic.from_json"
        warnings.warn(msg, FutureWarning, stacklevel=2)

        # make sure dims is a list/tuple
        if not isinstance(dims, (tuple, list)):
            raise TypeError("`dims` should be a list or tuple of strings")

        # read from file
        try:
            if len(dims) == 1:
                data, meta = _Read1DPlainText(filename)
            elif len(dims) == 2:
                data, meta = _Read2DPlainText(filename)
        except Exception as e:
            msg = "unable to read plaintext file, perhaps the dimension of the file does "
            msg += "not match the passed `dims`;\nexception: %s" %str(e)
            raise ValueError(msg)

        # get the bin edges
        edges = meta.pop('edges', None)
        if edges is None:
            raise ValueError("plaintext file does not include `edges`; cannot be loaded into a BinnedStatistic")
        if len(dims) == 1:
            edges = [edges]

        meta.update(kwargs)
        return cls(dims, edges, data, **meta)


    def copy(self, cls=None):
        """
        Returns a copy of the BinnedStatistic, optionally change the type
        to cls. cls must be a subclass of BinnedStatistic.
        """
        attrs = self.__copy_attrs__()
        if cls is None:
            cls = self.__class__
        if not issubclass(cls, BinnedStatistic):
            raise TypeError("The cls argument must be a subclass of BinnedStatistic")

        return cls.__construct_direct__(self.data.copy(), self.mask.copy(), **attrs)

    def rename_variable(self, old_name, new_name):
        """
        Rename a variable in :attr:`data` from ``old_name`` to ``new_name``.

        Note that this procedure is performed in-place (does not
        return a new BinnedStatistic)

        Parameters
        ----------
        old_name : str
            the name of the old varibale to rename
        new_name : str
            the desired new variable name

        Raises
        ------
        ValueError
            If `old_name` is not present in :attr:`variables`
        """
        import copy
        if old_name not in self.variables:
            raise ValueError("`%s` is not an existing variable name" %old_name)
        new_dtype = copy.deepcopy(self.data.dtype)

        names = list(new_dtype.names)
        names[names.index(old_name)] = new_name
        new_dtype.names = names
        self.data.dtype = new_dtype

    def sel(self, method=None, **indexers):
        """
        Return a new BinnedStatistic indexed by coordinate values along the
        specified dimension(s).

        Notes
        -----
        Scalar values used to index a specific dimension will result in
        that dimension being squeezed. To keep a dimension of unit length,
        use a list to index (see examples below).

        Parameters
        ----------
        method : {None, 'nearest'}
            The method to use for inexact matches; if set to `None`, require
            an exact coordinate match, otherwise match the nearest coordinate
        **indexers :
            the pairs of dimension name and coordinate value used to index
            the BinnedStatistic

        Returns
        -------
        sliced : BinnedStatistic
            a new BinnedStatistic holding the sliced data and coordinate grid

        Examples
        --------
        >>> pkmu
        <BinnedStatistic: dims: (k: 200, mu: 5), variables: ('mu', 'k', 'power')>

        >>> pkmu.sel(k=0.4)
        <BinnedStatistic: dims: (mu: 5), variables: ('mu', 'k', 'power')>

        >>> pkmu.sel(k=[0.4])
        <BinnedStatistic: dims: (k: 1, mu: 5), variables: ('mu', 'k', 'power')>

        >>> pkmu.sel(k=slice(0.1, 0.4), mu=0.5)
        <BinnedStatistic: dims: (k: 30), variables: ('mu', 'k', 'power')>
        """
        indices = {}
        squeezed_dims = []
        for dim, key in indexers.items():

            if isinstance(key, list):
                indices[dim] = [self._get_index(dim, k, method=method) for k in key]
            elif isinstance(key, slice):
                new_slice = []
                for name in ['start', 'stop']:
                    new_slice.append(self._get_index(dim, getattr(key, name), method=method))
                i = self.dims.index(dim)
                indices[dim] = list(range(*slice(*new_slice).indices(self.shape[i])))
            elif not numpy.isscalar(key):
                raise IndexError("please index using a list, slice, or scalar value")
            else:
                indices[dim] = [self._get_index(dim, key, method=method)]
                squeezed_dims.append(dim)

        # can't squeeze all dimensions!!
        if len(squeezed_dims) == len(self.dims):
            raise IndexError("cannot return object with all remaining dimensions squeezed")

        toret = self.take(**indices)

        for dim in squeezed_dims:
            toret = toret.squeeze(dim)
        return toret

    def take(self, *masks, **indices):
        """
        Take a subset of a BinnedStatistic from given list of indices.
        This is more powerful but more verbose than `sel`. Also the
        result is never squeezed, even if only a single item along the direction
        is used.

        Parameters
        ----------
        masks : array_like (boolean)
            a list of masks that are of the same shape as the data.

        indices: dict (string : array_like)
            mapping from axes (by name, dim) to items to select (list/array_like).
            Each item is a valid selector for numpy's fancy indexing.

        Returns
        -------
        new BinnedStatistic, where only items selected by all axes are kept.

        Examples
        --------
        >>> pkmu
        <BinnedStatistic: dims: (k: 200, mu: 5), variables: ('mu', 'k', 'power')>

        # similar to pkmu.sel(k > 0.4), select the bin centers
        >>> pkmu.take(k=pkmu.coords['k'] > 0.4)
        <BinnedStatistic: dims: (mu: 5), variables: ('mu', 'k', 'power')>

        # also similar to pkmu.sel(k > 0.4), select the bin averages
        >>> pkmu.take(pkmu['k'] > 0.4)
        <BinnedStatistic: dims: (k: 30), variables: ('mu', 'k', 'power')>

        # impossible with sel.
        >>> pkmu.take(pkmu['modes'] > 0)

        """
        indices_dict = {}
        indices_dict.update(indices) # rename for a cleaner API.

        # flatten the masks, will keep items that are true everywhere
        mask = numpy.ones(self.shape, dtype='?')
        for m in masks: mask = mask & m

        indices = [numpy.ones(self.shape[i], dtype='?') for i in range(len(self.dims))]

        # update indices with masks
        for i, dim in enumerate(self.dims):
            axis = list(range(len(self.dims)))
            axis.remove(i)
            axis = tuple(axis)
            mask1 = mask.all(axis=axis)
            indices[i] &= mask1

        # update indices with indices_dict
        for dim, index in indices_dict.items():
            i = self.dims.index(dim)
            if isinstance(index, numpy.ndarray) and index.dtype == numpy.dtype('?'):
                # boolean mask?
                assert index.ndim == 1
                indices[i] &= index
            else:
                mask1 = numpy.zeros(self.shape[i], dtype='?')
                mask1.put(index, True)
                indices[i] &= mask1

        # convert to indices
        for i in range(len(indices)):
            indices[i] = indices[i].nonzero()[0]

        data = self.data.copy()
        mask = self.mask.copy()
        for i, idx in enumerate(indices):
            data = numpy.take(data, idx, axis=i)
            mask = numpy.take(mask, idx, axis=i)

        toret = self.__finalize__(data, mask, indices)

        return toret

    def squeeze(self, dim=None):
        """
        Squeeze the BinnedStatistic along the specified dimension, which
        removes that dimension from the BinnedStatistic.

        The behavior is similar to that of :func:`numpy.squeeze`.

        Parameters
        ----------
        dim : str, optional
            The name of the dimension to squeeze. If no dimension
            is provided, then the one dimension with unit length will
            be squeezed

        Returns
        -------
        squeezed : BinnedStatistic
            a new BinnedStatistic instance, squeezed along one dimension

        Raises
        ------
        ValueError
            If the specified dimension does not have length one, or
            no dimension is specified and multiple dimensions have
            length one

        Examples
        --------
        >>> pkmu
        <BinnedStatistic: dims: (k: 200, mu: 1), variables: ('mu', 'k', 'power')>
        >>> pkmu.squeeze() # squeeze the mu dimension
        <BinnedStatistic: dims: (k: 200), variables: ('mu', 'k', 'power')>
        """
        # infer the right dimension to squeeze
        if dim is None:
            dim = [k for k in self.dims if len(self.coords[k]) == 1]
            if not len(dim):
                raise ValueError("no available dimensions with length one to squeeze")
            if len(dim) > 1:
                raise ValueError("multiple dimensions available to squeeze -- please specify")
            dim = dim[0]
        else:
            if dim not in self.dims:
                raise ValueError("`%s` is not a valid dimension name" %dim)
            if len(self.coords[dim]) != 1:
                raise ValueError("the `%s` dimension must have length one to squeeze" %dim)

        # remove the dimension from the grid
        i = self.dims.index(dim)
        toret = self.copy()
        toret.dims.pop(i); toret.edges.pop(dim); toret.coords.pop(dim)
        if not len(toret.dims):
            raise ValueError("cannot squeeze the only remaining axis")

        # construct new object with squeezed data/mask
        attrs = toret.__copy_attrs__()
        d = toret.data.squeeze(axis=i)
        m = toret.mask.squeeze(axis=i)
        return self.__construct_direct__(d, m, **attrs)

    def average(self, dim, **kwargs):
        """
        Compute the average of each variable over the specified dimension.

        Parameters
        ----------
        dim : str
            The name of the dimension to average over
        **kwargs :
            Additional keywords to pass to :func:`BinnedStatistic.reindex`. See the
            documentation for :func:`BinnedStatistic.reindex` for valid keywords.

        Returns
        -------
        averaged : BinnedStatistic
            A new BinnedStatistic, with data averaged along one dimension,
            which reduces the number of dimension by one
        """
        spacing = (self.edges[dim][-1] - self.edges[dim][0])
        toret = self.reindex(dim, spacing, **kwargs)
        return toret.sel(**{dim:toret.coords[dim][0]})

    def reindex(self,
                    dim,
                    spacing,
                    weights=None,
                    force=True,
                    return_spacing=False,
                    fields_to_sum=[]):
        """
        Reindex the dimension ``dim`` by averaging over multiple coordinate bins,
        optionally weighting by ``weights``.

        Returns a new BinnedStatistic holding the re-binned data.

        Notes
        -----
        *   We can only re-bin to an integral factor of the current
            dimension size in order to inaccuracies when re-binning to
            overlapping bins
        *   Variables specified in `fields_to_sum` will
            be summed when re-indexing, instead of averaging


        Parameters
        ----------
        dim : str
            The name of the dimension to average over
        spacing : float
            The desired spacing for the re-binned data. If `force = True`,
            the spacing used will be the closest value to this value, such
            that the new bins are N times larger, when N is an integer
        weights : array_like or str, optional (`None`)
            An array to weight the data by before re-binning, or if
            a string is provided, the name of a data column to use as weights
        force : bool, optional
            If `True`, force the spacing to be a value such
            that the new bins are N times larger, when N is an integer,
            otherwise, raise an exception. Default is `True`
        return_spacing : bool, optional
            If `True`, return the new spacing as the second return value.
            Default is `False`.
        fields_to_sum : list
            the name of fields that will be summed when reindexing, instead
            of averaging

        Returns
        -------
        rebinned : BinnedStatistic
            A new BinnedStatistic instance, which holds the rebinned coordinate
            grid and data variables
        spacing : float, optional
            If `return_spacing` is `True`, the new coordinate spacing
            will be returned
        """
        i = self.dims.index(dim)
        fields_to_sum += self._fields_to_sum

        # determine the new binning
        old_spacings = numpy.diff(self.coords[dim])
        if not numpy.array_equal(old_spacings, old_spacings):
            raise ValueError("`reindex` requires even bin spacings")
        old_spacing = old_spacings[0]

        factor = numpy.round(spacing/old_spacing).astype('int')
        if not factor:
            raise ValueError("new spacing must be smaller than original spacing of %.2e" %old_spacing)
        if factor == 1:
            raise ValueError("closest binning size to input spacing is the same as current binning")
        if not numpy.allclose(old_spacing*factor, spacing) and not force:
            raise ValueError("if `force = False`, new bin spacing must be an integral factor smaller than original")

        # make a copy of the data
        data = self.data.copy()

        # get the weights
        if isinstance(weights, str):
            if weights not in self.variables:
                raise ValueError("cannot weight by `%s`; no such column" %weights)
            weights = self.data[weights]

        edges = self.edges[dim]
        new_shape = list(self.shape)

        # check if we need to discard bins from the end
        leftover = self.shape[i] % factor
        if leftover and not force:
            args = (leftover, old_spacing*factor)
            raise ValueError("cannot re-bin because they are %d extra bins, using spacing = %.2e" %args)
        if leftover:
            sl = [slice(None, None)]*len(self.dims)
            sl[i] = slice(None, -leftover)
            data = data[sl]
            if weights is not None: weights = weights[sl]
            edges = edges[:-leftover]
            new_shape[i] = new_shape[i] - leftover

        # new edges
        new_shape[i] /= factor
        new_shape[i] = int(new_shape[i])
        new_edges = numpy.linspace(edges[0], edges[-1], new_shape[i]+1)

        # the re-binned data
        new_data = numpy.empty(new_shape, dtype=self.data.dtype)
        for name in self.variables:
            operation = numpy.nanmean
            weights_ = weights
            if weights is not None or name in fields_to_sum:
                operation = numpy.nansum
                if name in fields_to_sum: weights_ = None
            new_data[name] = bin_ndarray(data[name], new_shape, weights=weights_, operation=operation)

        # the new mask
        new_mask = numpy.zeros_like(new_data, dtype=bool)
        for name in self.variables:
            new_mask = numpy.logical_or(new_mask, ~numpy.isfinite(new_data[name]))

        # construct new object
        kw = self.__copy_attrs__()
        kw['edges'][dim] = new_edges
        kw['coords'][dim] = 0.5*(new_edges[1:] + new_edges[:-1])
        toret = self.__construct_direct__(new_data, new_mask, **kw)

        return (toret, spacing) if return_spacing else toret

#------------------------------------------------------------------------------
# Deprecated Plaintext read/write functions
#------------------------------------------------------------------------------
def _Read2DPlainText(filename):
    """
    Reads the plain text storage of a 2D measurement

    Returns
    -------
    data : dict
        dictionary holding the `edges` data, as well as the
        data columns for the P(k,mu) measurement
    metadata : dict
        any additional metadata to store as part of the
        P(k,mu) measurement
    """
    d = {}
    metadata = {}

    with open(filename, 'r') as ff:

        # read number of k and mu bins are first line
        Nk, Nmu = [int(l) for l in ff.readline().split()]
        N = Nk*Nmu

        # names of data columns on second line
        columns = ff.readline().split()

        lines = ff.readlines()
        data = numpy.array([float(l) for line in lines[:N] for l in line.split()])
        data = data.reshape((Nk, Nmu, -1)) #reshape properly to (Nk, Nmu)

        # make a dict, making complex arrays from real/imag parts
        i = 0
        while i < len(columns):
            name = columns[i]
            nextname = columns[i+1] if i < len(columns)-1 else ''
            if name.endswith('.real') and nextname.endswith('.imag'):
                name = name.split('.real')[0]
                d[name] = data[...,i] + 1j*data[...,i+1]
                i += 2
            else:
                d[name] = data[...,i]
                i += 1

        # store variables as a structured array
        dtypes = numpy.dtype([(name, d[name].dtype) for name in d])
        data = numpy.empty(data.shape[:2], dtype=dtypes)
        for name in d:
            data[name] = d[name]

        # read the edges for k and mu bins
        edges = []
        l1 = int(lines[N].split()[-1]); N = N+1
        edges.append(numpy.array([float(line) for line in lines[N:N+l1]]))
        l2 = int(lines[N+l1].split()[-1]); N = N+l1+1
        edges.append(numpy.array([float(line) for line in lines[N:N+l2]]))
        metadata['edges'] = edges

        # read any metadata
        if len(lines) > N+l2:
            N_meta = int(lines[N+l2].split()[-1])
            N = N + l2 + 1
            meta = lines[N:N+N_meta]
            for line in meta:
                fields = line.split()
                cast = fields[-1]
                if cast in __builtins__:
                    metadata[fields[0]] = __builtins__[cast](fields[1])
                elif hasattr(numpy, cast):
                     metadata[fields[0]] = getattr(numpy, cast)(fields[1])
                else:
                    raise TypeError("Metadata must have builtin or numpy type")

    return data, metadata

def _Read1DPlainText(filename):
    """
    Reads the plain text storage of a 1D measurement

    Notes
    -----
    *   If `edges` is present in the file, they will be returned
        as part of the metadata, with the key `edges`
    *   If the first line of the file specifies column names,
        they will be returned as part of the metadata with the
        `columns` key

    Returns
    -------
    data : array_like
        the 1D data stacked vertically, such that each columns
        represents a separate data variable
    metadata : dict
        any additional metadata to store as part of the
        P(k) measurement
    """
    # data list
    data = []

    # extract the metadata
    metadata = {}
    make_float = lambda x: float(x[1:])
    with open(filename, 'r') as ff:

        currline = 0
        lines = ff.readlines()

        # try to read columns
        if lines[0][0] == '#':
            try:
                metadata['columns'] = lines[0][1:].split()
            except:
                pass

        while True:

            # break if we are at the EOF
            if currline == len(lines): break
            line = lines[currline]

            if not line:
                currline += 1
                continue

            if line[0] != '#':
                data.append([float(l) for l in line.split()])
            else:
                line = line[1:]

                # read edges
                if 'edges' in line:
                    fields = line.split()
                    N = int(fields[-1]) # number of edges
                    metadata['edges'] = numpy.array([make_float(l) for l in lines[currline+1:currline+1+N]])
                    currline += 1+N
                    continue

                # read metadata
                if 'metadata' in line:
                    # read and cast the metadata properly
                    fields = line.split()
                    N = int(fields[-1]) # number of individual metadata lines
                    for i in range(N):
                        fields = lines[currline+1+i][1:].split()
                        cast = fields[-1]
                        if cast in __builtins__:
                            metadata[fields[0]] = __builtins__[cast](fields[1])
                        elif hasattr(numpy, cast):
                             metadata[fields[0]] = getattr(numpy, cast)(fields[1])
                        else:
                            raise TypeError("metadata must have builtin or numpy type")
                    currline += 1+N
                    continue

            # add to the data
            currline += 1

    data = numpy.asarray(data)

    # get names of columns, using default if not in file
    columns = metadata.pop('columns', ['col_%d' %i for i in range(data.shape[1])])

    # make a dict, making complex arrays from real/imag parts
    i = 0
    d = {}
    while i < len(columns):
        name = columns[i]
        nextname = columns[i+1] if i < len(columns)-1 else ''
        if name.endswith('.real') and nextname.endswith('.imag'):
            name = name.split('.real')[0]
            d[name] = data[...,i] + 1j*data[...,i+1]
            i += 2
        else:
            d[name] = data[...,i]
            i += 1

    # store variables as a structured array
    dtypes = numpy.dtype([(name, d[name].dtype) for name in d])
    data = numpy.empty(len(data), dtype=dtypes)
    for name in d:
        data[name] = d[name]

    return data, metadata
