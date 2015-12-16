

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
    
    Example
    -------
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
    
    
class DataSet(object):
    """
    Lightweight class to hold variables at fixed coordinates, i.e.,
    a grid of (r,mu) or (k,mu) bins 
    
    It is modeled after the syntax of ``xray.DataSet``, and is designed 
    to hold correlation function or power spectrum results (in 1D or 2D)
    
    
    Attributes
    ----------
    data    : array_like
        a structured array holding the data variables on the coordinate grid
        
    mask    : array_like
        a boolean array where `True` elements indicate that that coordinate
        grid point has a data variable that is `inf` or `NaN`
        
    attrs   : collections.OrderedDict
        an ordered dictionary holding any meta data that has been attached
        to the instance
        
    dims    : list
        a list of strings specifying the names of each axis of the input data
    
    edges   : dict
         a dict holding the bin edges for each dimension in `dims`
        
    coords : dict
        a dict holding the bin centers for each dimension in `dims`
    
    force_index_match : bool
        if `True`, when indexing using coordinate values, return
        results for the nearest bin to the value specified
        
    sum_only : list of str
        a list of strings specifying fields in `data` that will 
        only be summed when combining bins.
        
    ndim : int
        the number of dimensions
    
    shape : tuple
        a tuple holding the shape of the data variables
    
    variables : list
        the list of names of each data variable
        
    Examples
    --------
    The following example shows how to read a power.py 2d output into
    a PkmuResult object.
    
    >>> from nbodykit import files
    >>> d, meta = files.ReadPower2DPlainText('some2dfile.txt')
    >>> pkmuobj = PkmuResult.from_dict(d, **meta)
    """
    def __init__(self, dims, edges, variables, 
                    force_index_match=False, sum_only=[], **kwargs):
        """
        Parameters
        ----------
        dims : list, (Ndim,)
            a list of strings specifying names for the coordinate dimensions.
            The suffix '_cen' will be appended to each dimension name, to indicate
            that the coordinate grid is defined at the bin centers
            
        edges : list, (Ndim,)
            a list specifying the bin edges for each dimension
            
        variables : dict
            a dictionary holding the data variables, where the keys
            are interpreted as the variable names
            
        force_index_match : bool, optional
             if `True`, when indexing using coordinate values, 
             automatically return the results for the closest coordinate
             to the input values
             
        sum_only : list
            a list of strings specifying variables that will 
            only be summed (not averaged) when combining bins
            
        **kwargs :
            any additional keywords are saved as metadata in the ``attrs``
            attribute of the class
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
        dtypes = [(name, variables[name].dtype) for name in variables]            
        self.data = numpy.empty(self.shape, dtype=numpy.dtype(dtypes))
        for name in variables:
            self.data[name] = variables[name]
            
        # define a mask such that a coordinate grid element will be masked
        # if any of the variables at that coordinate are (NaN, inf)
        self.mask = numpy.zeros(self.shape, dtype=bool)
        for name in variables:
            self.mask = numpy.logical_or(self.mask, ~numpy.isfinite(self.data[name]))
             
        # If `True`: always returns nearest bin value
        self.force_index_match = force_index_match
        
        # variables which are not averaged
        self.sum_only = sum_only
        
        # save track metadata
        self.attrs = OrderedDict()
        for k in kwargs:
            self.attrs[k] = kwargs[k]
            
    @property
    def ndim(self):
        """
        The number of coordinate dimensions
        """
        return len(self.dims)
        
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
        return a new class instance
        """
        obj = object.__new__(cls)
        for k in kwargs:
            v = kwargs[k] if not hasattr(kwargs[k], 'copy') else kwargs[k].copy()
            setattr(obj, k, v)
            
        for k, d in zip(['data', 'mask'], [data, mask]):
            setattr(obj, k, d)
            if obj.shape != d.shape:
                setattr(obj, k, d.reshape(obj.shape))

        return obj

    def __finalize__(self, data, mask, indices):
        """
        Finalize and return a new instance from a slice of the 
        current object (returns a copy)
        """
        edges, coords = self.__slice_edges__(indices)
        
        kw = {'dims':list(self.dims), 'edges':edges, 'coords':coords, 'attrs':self.attrs.copy()}
        kw['sum_only'] = list(self.sum_only)
        kw['force_index_match'] = self.force_index_match
        
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
            
        dtype = self.data.dtype.descr
        if key not in self.data.dtype.names:
            dtype += [(key, data.dtype.type)]
            
        new = numpy.zeros(self.data.shape, dtype=dtype)
        for col in self.variables:
            new[col] = self.data[col]
            
        new[key] = data
        mask = numpy.logical_or(self.mask, ~numpy.isfinite(new[key]))
        self.data = new
        self.mask = mask
        
    def __getitem__(self, key):
        """
        If a string key is supplied, return the associated
        `variable` or `coord` value with that dimension name
        
        Otherwise, if array indexing is supplied, slice the 
        `data` attribute, returning a new class instance that 
        holds the sliced coordinate grid
        """
        if key in self.variables:
            return self.data[key]
        elif key in self.coords.keys():
            return self.coords[key]
        
        key_ = key
        if isinstance(key, int) or all(isinstance(x, int) for x in key):
            key_ = [key]

        indices = [range(0, self.shape[i]) for i in range(self.ndim)]
        for i, subkey in enumerate(key_):
            if isinstance(subkey, int):
                indices[i] = [subkey]
            elif isinstance(subkey, list):
                indices[i] = subkey
            elif isinstance(subkey, slice):
                indices[i] = range(*subkey.indices(self.shape[i]))
                
        return self.__finalize__(self.data[key], self.mask[key], indices)
        
    def _get_index(self, dim, val):
        """
        Internal function to compute the bin index of the nearest 
        coordinate value to the input value
        """
        index = self.coords[dim]
        if self.force_index_match:
            i = (numpy.abs(index-val)).argmin()
        else:
            try:
                i = list(index).index(val)
            except Exception as e:
                args = (dim, str(e))
                msg = "error converting '%s' index; try setting `force_index_match=True`: %s"
                raise IndexError(msg %args)
                
        return i
        
    #--------------------------------------------------------------------------
    # user-called functions
    #--------------------------------------------------------------------------
    def sel(self, **kwargs):
        """
        Label-based indexing by coordinate name. Indexing should be supplied
        using the dimension name as the key and the desired coordinate values as
        the keyword value.
        
        Returns
        -------
        sliced : DataSet
            a new ``DataSet`` holding the sliced data and coordinate grid
        
        
        Examples
        --------
        >>> pkmu
        <DataSet: dims: (k_cen: 200, mu_cen: 5), variables: ('mu', 'k', 'power')>
        >>> sliced_1 = pkmu.sel(k_cen=0.4)
        >>> sliced_1
        <DataSet: dims: (k_cen: 1, mu_cen: 5), variables: ('mu', 'k', 'power')>
        >>> sliced_2 = pkmu.sel(k_cen=slice(0.1, 0.4), mu_cen=0.5)
        >>> sliced_2
        <DataSet: dims: (k_cen: 30, mu_cen: 1), variables: ('mu', 'k', 'power')>
        """
        indices = [range(0, self.shape[i]) for i in range(self.ndim)]
        for dim in kwargs:
            key = kwargs[dim]
            i = self.dims.index(dim)
            
            if isinstance(key, list):
                indices[i] = [self._get_index(dim, k) for k in key]
            elif isinstance(key, slice):
                new_slice = []
                for name in ['start', 'stop']:
                    new_slice.append(self._get_index(dim, getattr(key, name)))
                indices[i] = range(*slice(*new_slice).indices(self.shape[i]))
            else:
                indices[i] = [self._get_index(dim, key)]
        
        # check for empty slices
        for i, idx in enumerate(indices):
            if not len(idx):
                raise KeyError("trying to use empty slice for dimension '%s'" %self.dims[i])
        
        data = self.data.copy()
        mask = self.mask.copy()
        for i, idx in enumerate(indices):
            data = numpy.take(data, idx, axis=i)
            mask = numpy.take(mask, idx, axis=i)
        
        return self.__finalize__(data, mask, indices)
        
    def to_pickle(self, filename):
        """
        Dump the object to the specified file as a pickle
        
        Parameters
        ----------
        filename : str
            the name of the file holding the pickle
        """
        necessary = ['data', 'mask', 'dims', 'edges', 'coords', 
                    'attrs', 'force_index_match', 'sum_only']
        d = {k:getattr(self,k) for k in necessary}
        pickle.dump(d, open(filename, 'w'))
        
    @classmethod
    def from_pickle(cls, filename):
        """
        Read a `DataSet` from a pickle, assuming the pickle was 
        created with `DataSet.to_pickle`
        
        Parameters
        ----------
        filename : str
            the name of the pickle file to read from
        """
        d = pickle.load(open(filename, 'r'))
        data, mask = d.pop('data'), d.pop('mask')
        return cls.__construct_direct__(data, mask, **d)
        
    @classmethod
    def from_nbkit(cls, d, meta):
        """
        Return a DataSet object from a dictionary of data. Additional
        metadata can be specified as keyword arguments
        
        Notes
        -----
        * the dictionary ``d`` must also have entries for ``dims``
        and ``edges`` used to construct the ``DataSet``
        
        Parameters
        ----------
        d : dict 
            a dictionary holding the data variables, as well as the
            `dims` and `edges` values
        meta : dict
            dictionary of metadata to store in the ``attrs`` attribute
        """
        d = d.copy() # copy so we don't edit for caller
        for name in ['dims', 'edges']:
            if name not in d:
                raise ValueError("must supply `%s` value in input dictionary" %name)
                
        edges = d.pop('edges')
        dims = d.pop('dims')
        return cls(dims, edges, d, **meta)
                
    def squeeze(self, dim, value):
        """
        Select the data along the specified dimension with input coordinate
        value, returning a squeezed `DataSet`, which has the dimension that
        we selected along removed
        
        Parameters
        ----------
        dim : str
            the name of the dimension to squeeze
        value : int, float
            the coordinate value to select along the `dim` axis
        
        Returns
        -------
        squeezed : DataSet
            a new `DataSet` instance, squeezed along one dimension
        
        Examples
        --------
        >>> pkmu
        <DataSet: dims: (k_cen: 200, mu_cen: 5), variables: ('mu', 'k', 'power')>
        >>> pkmu.squeeze('k_cen', 0.4)
        <DataSet: dims: (mu_cen: 5), variables: ('mu', 'k', 'power')>
        """
        i = self._get_index(dim, value)
        idx = [slice(None, None)]*self.ndim
        idx[self.dims.index(dim)] = i
        
        toret = self[idx]        
        toret.dims.pop(self.dims.index(dim))
        toret.edges.pop(dim)
        toret.coords.pop(dim)
        kw = {'dims':toret.dims, 'edges':toret.edges, 'coords':toret.coords, 'attrs':toret.attrs}
        kw['sum_only'] = self.sum_only
        kw['force_index_match'] = self.force_index_match

        return self.__construct_direct__(toret.data.squeeze(), toret.mask.squeeze(), **kw)
        
    def average(self, dim, weights=None):
        """
        Compute the average of each variable over the specified dimension, 
        optionally using `weights`
        
        Parameters
        ----------
        dim : str
            the name of the dimension to average over
        
        Returns
        -------
        averaged : DataSet
            a new `DataSet` instance, averaged along one dimension, such
            that there is one less dimension now
        """
        spacing = (self.edges[dim][-1] - self.edges[dim][0])
        toret = self.reindex(dim, spacing, weights=weights)
        return toret.squeeze(dim, toret.coords[dim][0])
        
    def reindex(self, dim, spacing, weights=None, force=True, return_spacing=False):
        """
        Reindex the dimension `dim` and return a new `DataSet` holding
        the re-binned data, optionally weighted by `weights`. 
        
        Notes
        -----
        We can only re-bin to an integral factor of the current 
        dimension size in order to inaccuracies when re-binning to 
        overlapping bins
        
        
        Parameters
        ----------
        dim : str
            the name of the dimension to average over
        spacing : float
            the desired spacing for the re-binned data. If `force = True`,
            the spacing used will be the closest value to this value, such
            that the new bins are N times larger, when N is an integer
        weights : array_like or str, optional (`None`)
            an array to weight the data by before re-binning, or if
            a string is provided, the name of a data column to use as weights
        force : bool, optional (`True`)
            if `True`, force the spacing to be a value such
            that the new bins are N times larger, when N is an integer; otherwise,
            raise an exception
        return_spacing : bool, optional (`False`)
            if `True`, return the new k spacing as the second return value
            
        Returns
        -------
        rebinned : DataSet
            a new `DataSet` instance, which holds the rebinned coordinate
            grid and data variables
        """        
        i = self.dims.index(dim)
        
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
        if old_spacing*factor != spacing and not force: 
            raise ValueError("if `force = False`, new bin spacing must be an integral factor smaller than original")
        
        # make a copy of the data
        data = self.data.copy()
                
        # get the weights
        if isinstance(weights, str):
            if weights not in self.variables:
                raise ValueError("cannot weight by `%s`; no such column" %weights)
            weights = self.data[weights]
            
        edges = self.edges[dim]
        
        # check if we need to discard bins from the end
        leftover = self.shape[i] % factor
        if leftover and not force:
            args = (leftover, old_spacing*factor)
            raise ValueError("cannot re-bin because they are %d extra bins, using spacing = %.2e" %args)
        if leftover:
            sl = [slice(None, None)]*self.ndim
            sl[i] = slice(None, -1)
            data = data[sl]
            if weights is not None: weights = weights[sl]
            edges = edges[:-1]
            
        # new edges
        new_shape = list(self.shape)
        new_shape[i] /= factor
        new_edges = numpy.linspace(edges[0], edges[-1], new_shape[i]+1)
        
        # the re-binned data
        new_data = numpy.empty(new_shape, dtype=self.data.dtype)
        for name in self.variables:
            operation = numpy.nanmean
            weights_ = weights
            if weights is not None or name in self.sum_only:
                operation = numpy.nansum
                if name in self.sum_only: weights_ = None
            new_data[name] = bin_ndarray(data[name], new_shape, weights=weights_, operation=operation)
        
        # the new mask
        new_mask = numpy.zeros_like(new_data, dtype=bool)
        for name in self.variables:
            new_mask = numpy.logical_or(new_mask, ~numpy.isfinite(new_data[name]))
        
        # construct new object
        kw = {'dims':list(self.dims), 'edges':self.edges.copy(), 'coords':self.coords.copy(), 
                'attrs':self.attrs, 'sum_only':self.sum_only, 'force_index_match':self.force_index_match}
        kw['edges'][dim] = new_edges
        kw['coords'][dim] = 0.5*(new_edges[1:] + new_edges[:-1])
        toret = self.__construct_direct__(new_data, new_mask, **kw)
        
        return (toret, spacing) if return_spacing else toret
        
        

def from_1d_measurement(dims, d, meta, columns=None, **kwargs):
    """
    Return a DataSet object from columns of data and
    any additional meta data
    """
    if columns is None: 
        columns = meta.get('columns', None)
    
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
        raise ValueError("must supply `edges` value in `d` input" %name)

    d['edges'] = d.pop('edges')
    d['dims'] = dims
    meta.update(kwargs)
    return DataSet.from_nbkit(d, meta)


class Power1DDataSet(DataSet):
    """
    A `DataSet` that holds a 1D power spectrum in bins of `k`
    """
    def __init__(self, edges, variables, 
                    force_index_match=False, sum_only=[], **kwargs):
            
        dims = ['k']
        edges = [edges]
        kwargs['force_index_match'] = force_index_match
        kwargs['sum_only'] = sum_only
        super(Power1DDataSet, self).__init__(dims, edges, variables, **kwargs)
        
    @classmethod
    def from_nbkit(cls, d, meta, columns=None, **kwargs):
        """
        Return a `Power1DDataSet` instance taking the return values
        of `files.Read1DPlaintext` as input
               
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
        >>> power = Power1DDataSet(*files.Read1DPlaintext(filename),force_index_match=True)
        """
        toret = from_1d_measurement(['k'], d, meta, columns=columns, **kwargs)
        toret.__class__ = cls
        return toret
        
class Power2DDataSet(DataSet):
    """
    A `DataSet` that holds a 2D power spectrum in bins of `k` and `mu`
    """
    def __init__(self, edges, variables, 
                    force_index_match=False, sum_only=[], **kwargs):
            
        dims = ['k', 'mu']
        kwargs['force_index_match'] = force_index_match
        kwargs['sum_only'] = sum_only
        super(Power2DDataSet, self).__init__(dims, edges, variables, **kwargs)
        
    @classmethod
    def from_nbkit(cls, d, meta, **kwargs):
        """
        Return a `Power2DDataSet` instance taking the return values
        of `files.Read2DPlaintext` as input
               
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
        >>> power = Power2DDataSet(*files.Read2DPlaintext(filename),force_index_match=True)
        """
        toret = from_2d_measurement(['k', 'mu'], d, meta, **kwargs)
        toret.__class__ = cls
        return toret
        
class Corr1DDataSet(DataSet):
    """
    A `DataSet` that holds a 1D correlation function in bins of `r`
    """
    def __init__(self, edges, variables, 
                    force_index_match=False, sum_only=[], **kwargs):
                    
        dims = ['r']
        edges = [edges]
        kwargs['force_index_match'] = force_index_match
        kwargs['sum_only'] = sum_only
        super(Corr1DDataSet, self).__init__(dims, edges, variables, **kwargs)
        
    @classmethod
    def from_nbkit(cls, d, meta, columns=None, **kwargs):
        """
        Return a `Corr1DDataSet` instance taking the return values
        of `files.Read1DPlaintext` as input
               
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
        >>> corr = Power1DDataSet(*files.Read1DPlaintext(filename), force_index_match=True)
        """
        toret = from_1d_measurement(['r'], d, meta, columns=columns, **kwargs)
        toret.__class__ = cls
        return toret
        
class Corr2DDataSet(DataSet):
    """
    A `DataSet` that holds a 2D correlation in bins of `k` and `mu`
    """
    def __init__(self, edges, variables, 
                    force_index_match=False, sum_only=[], **kwargs):
            
        dims = ['r', 'mu']
        kwargs['force_index_match'] = force_index_match
        kwargs['sum_only'] = sum_only
        super(Corr2DDataSet, self).__init__(dims, edges, variables, **kwargs)
        
    @classmethod
    def from_nbkit(cls, d, meta, columns=None, **kwargs):
        """
        Return a `Corr2DDataSet` instance taking the return values
        of `files.Read2DPlaintext` as input
               
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
        >>> corr = Corr2DDataSet(*files.Read2DPlaintext(filename), force_index_match=True)
        """
        toret = from_2d_measurement(['r', 'mu'], d, meta, **kwargs)
        toret.__class__ = cls
        return toret
    
        


