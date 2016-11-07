import numpy
from collections import OrderedDict
import pickle

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

def __unpickle__(cls, d): 
    """
    Internal 'unpickling' function to create a new `cls` instance 
    from the dictionary `d`, which stores all the necessary information. 
    
    Notes
    -----
    *   Designed to be used with `DataSet.__reduce__`
    """
    data, mask = d.pop('data'), d.pop('mask')
    return cls.__construct_direct__(data, mask, **d)    
    
class DataSet(object):
    """
    Lightweight class to hold variables at fixed coordinates, i.e.,
    a grid of (r, mu) or (k, mu) bins for a correlation function or
    power spectrum measurement
    
    It is modeled after the syntax of :class:`xarray.Dataset`, and is designed 
    to hold correlation function or power spectrum results (in 1D or 2D)
    
    Notes
    -----
    *   the suffix *cen* will be appended to the names of the 
        dimensions passed to the constructor, since the :attr:`coords` array holds
        the **bin centers**, as constructed from the bin edges
        
    Examples
    --------
    The following example shows how to read a power spectrum or correlation 
    function measurement as written by a nbodykit `Algorithm`. It uses
    :func:`~nbodykit.files.Read1DPlainText`
    
    >>> from nbodykit import files
    >>> corr = Corr2dDataSet.from_nbkit(*files.Read2DPlainText(filename))
    >>> pk = Power1dDataSet.from_nbkit(*files.Read1DPlainText(filename))
        
    Data variables and coordinate arrays can be accessed in a dict-like
    fashion:
        
    >>> power = pkmu['power'] # returns power data variable
    >>> k_cen = pkmu['k_cen'] # returns k_cen coordinate array
        
    Array-like indexing of a :class:`DataSet` returns a new :class:`DataSet`
    holding the sliced data:
        
    >>> pkmu
    <DataSet: dims: (k_cen: 200, mu_cen: 5), variables: ('mu', 'k', 'power')>
    >>> pkmu[:,0] # select first mu column
    <DataSet: dims: (k_cen: 200), variables: ('mu', 'k', 'power')>
    
    Additional data variables can be added to the :class:`DataSet` via:
    
    >>> modes = numpy.ones((200, 5))
    >>> pkmu['modes'] = modes
    
    Coordinate-based indexing is possible through :func:`sel`:
    
    >>> pkmu
    <DataSet: dims: (k_cen: 200, mu_cen: 5), variables: ('mu', 'k', 'power')>
    >>> pkmu.sel(k_cen=slice(0.1, 0.4), mu_cen=0.5)
    <DataSet: dims: (k_cen: 30), variables: ('mu', 'k', 'power')>
    
    :func:`squeeze` will explicitly squeeze the specified dimension 
    (of length one) such that the resulting instance has one less dimension:
        
    >>> pkmu
    <DataSet: dims: (k_cen: 200, mu_cen: 1), variables: ('mu', 'k', 'power')>
    >>> pkmu.squeeze(dim='mu_cen') # can also just call pkmu.squeeze()
    <DataSet: dims: (k_cen: 200), variables: ('mu', 'k', 'power')>
    
    :func:`average` returns a new :class:`DataSet` holding the 
    data averaged over one dimension
        
    :func:`reindex` will re-bin the coordinate arrays along the specified 
    dimension
    """
    _fields_to_sum = []
    
    def __init__(self, dims, edges, variables, **kwargs):
        """
        Parameters
        ----------
        dims : list, (Ndim,)
            A list of strings specifying names for the coordinate dimensions.
            The dimension names stored in :attr:`dims` have the suffix 'cen'
            added, to indicate that the coordinate grid is defined at the bin 
            centers
            
        edges : list, (Ndim,)
            A list specifying the bin edges for each dimension
            
        variables : dict
            a dictionary holding the data variables, where the keys
            are interpreted as the variable names. The variable names are 
            stored in :attr:`variables`
            
        **kwargs :
            Any additional keywords are saved as metadata in the :attr:`attrs`
            attribute, which is an :class:`~collections.OrderedDict`
        """
        if len(dims) != len(edges):
            raise ValueError("size mismatch between specified `dims` and `edges`")
        
        shape = tuple(len(e)-1 for e in edges)
        for name in variables:
            if numpy.shape(variables[name]) != shape:
                args = (shape, numpy.shape(variables[name]))
                raise ValueError("`edges` imply data shape of %s, but data has shape %s" %args)
        
        self.dims = [dim+'_cen' for dim in dims]
        self.edges = dict(zip(self.dims, edges))
        
        # coordinates are the bin centers
        self.coords = {}
        for i, dim in enumerate(self.dims):
            self.coords[dim] = 0.5 * (edges[i][1:] + edges[i][:-1])
            
        # store variables as a structured array
        dtypes = numpy.dtype([(name, variables[name].dtype) for name in variables])
        self.data = numpy.empty(self.shape, dtype=dtypes)
        for name in variables:
            self.data[name] = variables[name]
            
        # define a mask such that a coordinate grid element will be masked
        # if any of the variables at that coordinate are (NaN, inf)
        self.mask = numpy.zeros(self.shape, dtype=bool)
        for name in variables:
            self.mask = numpy.logical_or(self.mask, ~numpy.isfinite(self.data[name]))
             
        # save and track metadata
        self.attrs = OrderedDict()
        for k in kwargs: self.attrs[k] = kwargs[k]
        
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
        the `DataSet`. This dictionary + `data` and `mask` are all
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
            idx = indices[i] + [indices[i][-1]+1]
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
        *   If a list of strings is passed, then a new `DataSet`
            holding only the `variable` names in `key` is returned
        *   Integer-based indexing or slices similar to numpy 
            indexing will slice `data`, returning a new 
            `DataSet` holding the newly sliced data and coordinate grid
        *   Scalar indexes (i.e., integers) used to index a certain
            dimension will "squeeze" that dimension, removing it
            from the coordinate grid
        """
        # if single string passed, return a coordinate or variable
        if isinstance(key, str):
            if key in self.variables:
                return self.data[key]
            elif key in self.coords.keys():
                return self.coords[key]
            else:
                raise KeyError("`%s` is not a valid variable or coordinate name" %key)
            
        # indices to slice the data with
        indices = [list(range(0, self.shape[i])) for i in range(len(self.dims))]
            
        # check for list/tuple of variable names
        # if so, return a DataSet with slice of columns
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
                raise IndexError("too many indices for DataSet; note that ndim = %d" %len(self.dims))
                
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
        
    def __reduce__(self):
        """
        The standard method for python pickle serialization
        """
        necessary = ['data', 'mask', 'dims', 'edges', 'coords', 'attrs']
        d = {k:getattr(self,k) for k in necessary}
        cls = self.__class__
        return (__unpickle__, (self.__class__, d, ))
        
    #--------------------------------------------------------------------------
    # user-called functions
    #--------------------------------------------------------------------------
    def copy(self):
        """
        Returns a copy of the DataSet
        """
        attrs = self.__copy_attrs__()
        cls = self.__class__
        return cls.__construct_direct__(self.data.copy(), self.mask.copy(), **attrs)
        
    def rename_variable(self, old_name, new_name):
        """
        Rename a variable in :attr:`data` from `old_name` to `new_name`
        
        Note that this procedure is performed in-place (does not 
        return a new DataSet)
        
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
        Return a new DataSet indexed by coordinate values along the 
        specified dimension(s) 
        
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
            the DataSet
            
        Returns
        -------
        sliced : DataSet
            a new DataSet holding the sliced data and coordinate grid
        
        Examples
        --------
        >>> pkmu
        <DataSet: dims: (k_cen: 200, mu_cen: 5), variables: ('mu', 'k', 'power')>
        
        >>> pkmu.sel(k_cen=0.4)
        <DataSet: dims: (mu_cen: 5), variables: ('mu', 'k', 'power')>
        
        >>> pkmu.sel(k_cen=[0.4])
        <DataSet: dims: (k_cen: 1, mu_cen: 5), variables: ('mu', 'k', 'power')>
        
        >>> pkmu.sel(k_cen=slice(0.1, 0.4), mu_cen=0.5)
        <DataSet: dims: (k_cen: 30), variables: ('mu', 'k', 'power')>
        """
        indices = [list(range(0, self.shape[i])) for i in range(len(self.dims))]
        squeezed_dims = []
        for dim in indexers:
            key = indexers[dim]
            i = self.dims.index(dim)
            
            if isinstance(key, list):
                indices[i] = [self._get_index(dim, k, method=method) for k in key]
            elif isinstance(key, slice):
                new_slice = []
                for name in ['start', 'stop']:
                    new_slice.append(self._get_index(dim, getattr(key, name), method=method))
                indices[i] = list(range(*slice(*new_slice).indices(self.shape[i])))
            elif not numpy.isscalar(key):
                raise IndexError("please index using a list, slice, or scalar value")
            else:
                indices[i] = [self._get_index(dim, key, method=method)]
                squeezed_dims.append(dim)
        
        # can't squeeze all dimensions!!
        if len(squeezed_dims) == len(self.dims):
            raise IndexError("cannot return object with all remaining dimensions squeezed")
        
        # check for empty slices
        for i, idx in enumerate(indices):
            if not len(idx):
                raise KeyError("trying to use empty slice for dimension '%s'" %self.dims[i])
        
        data = self.data.copy()
        mask = self.mask.copy()
        for i, idx in enumerate(indices):
            data = numpy.take(data, idx, axis=i)
            mask = numpy.take(mask, idx, axis=i)
        
        toret = self.__finalize__(data, mask, indices)
        for dim in squeezed_dims:
            toret = toret.squeeze(dim)
        return toret
                  
    @classmethod
    def from_nbkit(cls, d, meta):
        """
        Return a DataSet object from a dictionary of data and
        metadata
        
        Notes
        -----
        *   The dictionary `d` must also have entries for `dims`
            and `edges` which are used to construct the DataSet
        
        Parameters
        ----------
        d : dict 
            A dictionary holding the data variables, as well as the
            `dims` and `edges` values
        meta : dict
            dictionary of metadata to store in the :attr:`attrs` attribute
        
        Returns
        -------
        DataSet
            The newly constructed DataSet
        """
        d = d.copy() # copy so we don't edit for caller
        for name in ['dims', 'edges']:
            if name not in d:
                raise ValueError("must supply `%s` value in input dictionary" %name)
                
        edges = d.pop('edges')
        dims = d.pop('dims')
        return cls(dims, edges, d, **meta)
                
    def squeeze(self, dim=None):
        """
        Squeeze the DataSet along the specified dimension, which
        removes that dimension from the DataSet
        
        The behavior is similar to that of :func:`numpy.squeeze`.
        
        Parameters
        ----------
        dim : str, optional
            The name of the dimension to squeeze. If no dimension
            is provided, then the one dimension with unit length will 
            be squeezed
        
        Returns
        -------
        squeezed : DataSet
            a new DataSet instance, squeezed along one dimension
        
        Raises
        ------
        ValueError
            If the specified dimension does not have length one, or 
            no dimension is specified and multiple dimensions have 
            length one
        
        Examples
        --------
        >>> pkmu
        <DataSet: dims: (k_cen: 200, mu_cen: 1), variables: ('mu', 'k', 'power')>
        >>> pkmu.squeeze() # squeeze the mu dimension
        <DataSet: dims: (k_cen: 200), variables: ('mu', 'k', 'power')>
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
            Additional keywords to pass to :func:`DataSet.reindex`. See the 
            documentation for :func:`DataSet.reindex` for valid keywords.
        
        Returns
        -------
        averaged : DataSet
            A new DataSet, with data averaged along one dimension, 
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
        Reindex the dimension `dim` by averaging over multiple coordinate bins, 
        optionally weighting by `weights`. Return a new DataSet holding the 
        re-binned data
        
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
        rebinned : DataSet
            A new DataSet instance, which holds the rebinned coordinate
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
        
def from_1d_measurement(dims, d, meta, columns=None, **kwargs):
    """
    Return a DataSet object from columns of data and
    any additional meta data
    """
    meta = meta.copy() # make a copy
    if columns is None: 
        columns = meta.pop('columns', None)
    
    if 'edges' not in meta:
        raise ValueError("must supply `edges` value in `meta` input")
    if columns is None:
        raise ValueError("if `columns` key not in `meta`, then `columns` keyword must not be `None`")
    
    d = {col : d[:,i] for i, col in enumerate(columns)}
    d['edges'] = [meta.pop('edges')]
    d['dims'] = dims
    meta.update(kwargs)
    return DataSet.from_nbkit(d, meta)
    
def from_2d_measurement(dims, d, meta, **kwargs):
    """
    Return a DataSet object from a dictionary of data 
    and any additional data
    """
    if 'edges' not in d:
        raise ValueError("must supply `edges` value in `d` input")

    d['dims'] = dims
    meta.update(kwargs)
    return DataSet.from_nbkit(d, meta)


class Power1dDataSet(DataSet):
    """
    A `DataSet` that holds a 1D power spectrum in bins of `k`
    """
    _fields_to_sum = ['modes']
    
    def __init__(self, edges, variables, **kwargs):
        super(Power1dDataSet, self).__init__(['k'], [edges], variables, **kwargs)
        for field in ['modes']:
            if field in self: self._fields_to_sum.append(field)
        
    @classmethod
    def from_nbkit(cls, d, meta, columns=None, **kwargs):
        """
        Return a `Power1dDataSet` instance taking the return values
        of `files.Read1DPlainText` as input
               
        Parameters
        ----------
        d : array_like
            the 1D data stacked vertically, such that each columns
            represents a separate data variable
        meta : dict
            any additional metadata to store as part of the 
            P(k) measurement
        columns : list
            list of the column names -- required if `columns`
            not in `meta` dictionary
        
        Examples
        --------
        >>> from nbodykit import files
        >>> power = Power1dDataSet.from_nbkit(*files.Read1DPlainText(filename))
        """
        toret = from_1d_measurement(['k'], d, meta, columns=columns, **kwargs)
        toret.__class__ = cls
        return toret
        
class Power2dDataSet(DataSet):
    """
    A `DataSet` that holds a 2D power spectrum in bins of `k` and `mu`
    """
    _fields_to_sum = ['modes']
    
    def __init__(self, edges, variables, **kwargs):
        super(Power2dDataSet, self).__init__(['k','mu'], edges, variables, **kwargs)
        
    @classmethod
    def from_nbkit(cls, d, meta, **kwargs):
        """
        Return a `Power2dDataSet` instance taking the return values
        of `files.Read2DPlainText` as input
               
        Parameters
        ----------
        d : dict
            dictionary holding the `edges` data, as well as the
            data columns for the 2D measurement
        meta : dict
            any additional metadata to store as part of the 
            2D measurement
        
        Examples
        --------
        >>> from nbodykit import files
        >>> power = Power2dDataSet.from_nbkit(*files.Read2DPlainText(filename))
        """
        toret = from_2d_measurement(['k', 'mu'], d, meta, **kwargs)
        toret.__class__ = cls
        return toret
        
class Corr1dDataSet(DataSet):
    """
    A `DataSet` that holds a 1D correlation function in bins of `r`
    """
    _fields_to_sum = ['N', 'RR']
    
    def __init__(self, edges, variables, **kwargs):
        
        super(Corr1dDataSet, self).__init__(['r'], [edges], variables, **kwargs)
        for field in ['N', 'RR']:
            if field in self: self._fields_to_sum.append(field)
            
    @classmethod
    def from_nbkit(cls, d, meta, columns=None, **kwargs):
        """
        Return a `Corr1dDataSet` instance taking the return values
        of `files.Read1DPlainText` as input
               
        Parameters
        ----------
        d : array_like
            the 1D data stacked vertically, such that each columns
            represents a separate data variable
        meta : dict
            any additional metadata to store as part of the 
            P(k) measurement
        columns : list
            list of the column names -- required if `columns`
            not in `meta` dictionary
        
        Examples
        --------
        >>> from nbodykit import files
        >>> corr = Power1dDataSet.from_nbkit(*files.Read1DPlainText(filename))
        """
        toret = from_1d_measurement(['r'], d, meta, columns=columns, **kwargs)
        toret.__class__ = cls
        return toret
        
class Corr2dDataSet(DataSet):
    """
    A `DataSet` that holds a 2D correlation in bins of `k` and `mu`
    """
    _fields_to_sum = ['N', 'RR']
    
    def __init__(self, edges, variables, **kwargs):
        super(Corr2dDataSet, self).__init__(['r','mu'], edges, variables, **kwargs)
        
    @classmethod
    def from_nbkit(cls, d, meta, **kwargs):
        """
        Return a `Corr2dDataSet` instance taking the return values
        of `files.Read2DPlainText` as input
               
        Parameters
        ----------
        d : dict
            dictionary holding the `edges` data, as well as the
            data columns for the 2D measurement
        meta : dict
            any additional metadata to store as part of the 
            2D measurement
        
        Examples
        --------
        >>> from nbodykit import files
        >>> corr = Corr2dDataSet.from_nbkit(*files.Read2DPlainText(filename))
        """
        toret = from_2d_measurement(['r', 'mu'], d, meta, **kwargs)
        toret.__class__ = cls
        return toret
    
        


