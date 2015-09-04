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
    
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
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
    
class PkmuResult(object):
    """
    PkmuResult provides an interface to store and manipulate a 2D power 
    spectrum measurement, as a function of wavenumber `k` and line-of-sight
    angle `mu`. 
    
    Notes
    -----
    The `data` attribute can be accessed using the __getitem__ behavior
    of `PkmuResult`. Additionally, __getitem__ can be passed `k` and `mu`
    values, which can refer to the bin center values. If 
    `force_index_match = True`, then these values will automatically 
    match the nearest bin center value
    
    Attributes
    ----------
    data    : :py:class:`numpy.ma.core.MaskedArray`
        a masked structured array holding the data, where masked
        elements represent (k,mu) bins with missing data
    index   : :py:class:`numpy.ndarray`
        a structured array with field names `k_center`, `mu_center` that
        stores the center bin values. Same shape as `data`
    Nk      : int
        the number of k bins
    Nmu     : int
        the number of mu bins
    columns : list of str
        a list of the names of the fields in `data`
    kedges  : array_like
        the edges of the k bins used. shape is `Nk+1`
    muedges : array_like
        the edges of the mu bins used. shape is `Nmu+1`
    force_index_match : bool
        If `True`, when indexing using `k` or `mu` values, return
        results for the nearest bin to the value specified
    sum_only : list of str
        A list of strings specifying fields in `data` that will 
        only be summed when combining bins.
        
    Examples
    --------
    The following example shows how to read a power.py 2d output into
    a PkmuResult object.
    
    >>> from nbodykit import files
    >>> d, meta = files.ReadPower2DPlainText('some2dfile.txt')
    >>> pkmuobj = PkmuResult.from_dict(d, **meta)
    
    """
    def __init__(self, kedges, muedges, data, force_index_match=False, 
                  sum_only=[], **kwargs):
        """
        Parameters
        ----------
        kedges : array_like
            The list of the edges of the k bins
        muedges : array_like
            The list of the edges of the mu bins
        data : dict
            Dictionary holding 2D arrays of data. The keys
            will be used to infer the field names in the
            resulting numpy.recarray
        force_index_match : bool
             If `True`, when indexing using `k` or `mu` values, return
             results for the nearest bin to the value specified
        sum_only : list of str
            A list of strings specifying fields in `data` that will 
            only be summed when combining bins. 
        **kwargs
            Any additional metadata for the power spectrum object can
            be passed as keyword arguments here
        """
        self.kedges = kedges
        self.muedges = muedges
        columns = data.keys()
        
        # treat any NaNs as missing data
        mask = numpy.zeros((len(kedges)-1, len(muedges)-1), dtype=bool)
        dtypes = []
        for name in columns:
            mask = numpy.logical_or(mask, ~numpy.isfinite(data[name]))
            dtypes.append((name, data[name].dtype))

        # create a structured array to store the data
        shape = (len(self.kedges)-1, len(self.muedges)-1)
        self.data = numpy.empty(shape, dtype=numpy.dtype(dtypes))
        for name in columns:
            self.data[name] = data[name]
        
        # now make it a masked array
        self.data = numpy.ma.array(self.data, mask=mask)
        
        # now store the center (k,mu) of each bin as the index
        k_center = 0.5*(kedges[1:]+kedges[:-1])[...,None]
        mu_center = 0.5*(muedges[1:]+muedges[:-1])[None,...]
        dtypes = numpy.dtype([('k_center', k_center.dtype), ('mu_center', mu_center.dtype)])
        self.index = numpy.empty(shape, dtype=dtypes)
        self.index['k_center'], self.index['mu_center'] = numpy.broadcast_arrays(k_center,mu_center)

        # match closest index always returns nearest bin value
        self.force_index_match = force_index_match
        
        # fields which are averaged
        self.sum_only = sum_only
        
        # save any metadata too
        self._metadata = []
        for k, v in kwargs.iteritems():
            self._metadata.append(k)
            setattr(self, k, v)
    
    def __contains__(self, key):
        return key in self.columns
    
    def __getitem__(self, key):
        if key in self.columns:
            return self.data[key]
        try:
            new_key = ()
            
            # if tuple, check if we need to replace values with integers
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise IndexError("too many indices for array")
                
                for i, subkey in enumerate(key):
                    if isinstance(subkey, slice):
                        new_slice = []
                        for name in ['start', 'stop']:
                            val = getattr(subkey, name)
                            if not isinstance(val, int) and val is not None:
                                new_slice.append(self._get_index(i,val))
                            else:
                                new_slice.append(val)
                        new_key += (slice(*new_slice),)
                    elif not isinstance(subkey, int):
                        new_key += (self._get_index(i,subkey),)
                    else:
                        new_key += (subkey,)
                key = new_key            
                            
            return self.data[key]
        except Exception as e:
            raise KeyError("Key not understood in __getitem__: %s" %(str(e)))
    
    def to_pickle(self, filename):
        import pickle
        pickle.dump(self.__dict__, open(filename, 'w'))
        
    @classmethod
    def from_pickle(cls, filename):
        import pickle
        d = pickle.load(open(filename, 'r'))
        
        # the data
        data = {name : d['data'][name].data for name in d['columns']}
        
        # the metadata
        kwargs = {k:d[k] for k in d['_metadata']}
        
        # add the named keywords, if present
        kwargs['force_index_match'] = d.get('force_index_match', False)
        kwargs['sum_only'] = d.get('sum_only', [])
        return PkmuResult(d['kedges'], d['muedges'], data, **kwargs)
        
    @classmethod
    def from_dict(cls, d, **kwargs):
        """
        Return a PkmuResult object from a dictionary of data. Additional
        metadata can be specified as keyword arguments
        
        Notes
        -----
        The `edges` data must be given in the input dictionary. This should
        specify a list of size 2, with the format `[k_edges, mu_edges]`
        
        Parameters
        ----------
        d : dict
            dictionary of arrays to be stored as the data columns and bin 
            edges
        **kwargs 
            any additional keywords or metadata to be stored
        """
        d = d.copy() # copy so we don't edit for caller
        if 'edges' not in d:
            raise ValueError("must supply `edges` data in input dictionary")
        edges = d.pop('edges')
        return PkmuResult(edges[0], edges[1], d, **kwargs)
        
    @classmethod
    def from_list(cls, pkmus, weights=None, sum_only=[]):
        """
        Return an average PkmuResult object from a list of PkmuResult 
        objects, optionally using weights. For those columns in 
        sum_only, the data will be summed and not averaged.
        
        Notes
        -----
        The input power spectra must be defined on the same grid
        and have the same data columns.
        
        Parameters
        ----------
        pkmus : list of PkmuResult
            the list of 2D power spectrum objects to average
        
        """
        # check columns for all objects
        columns = [pkmu.columns for pkmu in pkmus]
        if not all(sorted(cols) == sorted(columns[0]) for cols in columns):
            raise ValueError("cannot combine PkmuResults with different column names")
        
        # check edges too
        for name in ['kedges', 'muedges']:
            edges = [getattr(pkmu,name) for pkmu in pkmus]
            if not all(numpy.allclose(e, edges[0]) for e in edges):
                raise ValueError("cannot combine PkmuResults with different %s" %name)
        
        # compute the weights
        if weights is None:
            weights = numpy.ones((len(pkmus), pkmus[0].Nk, pkmus[0].Nmu))
        else:
            if isinstance(weights, basestring):
                if weights not in columns[0]:
                    raise ValueError("Cannot weight by `%s`; no such column" %weights)
                weights = numpy.array([pkmu.data[weights].data for pkmu in pkmus])
        
        # take the mean or the sum
        data = {}    
        for name in columns[0]:
            col_data = numpy.array([pkmu.data[name] for pkmu in pkmus])
            if name not in sum_only:
                with numpy.errstate(invalid='ignore'):
                    data[name] = (col_data*weights).sum(axis=0) / weights.sum(axis=0)
            else:
                data[name] = numpy.sum(col_data, axis=0)
            
        # handle the metadata
        kwargs = {}
        for key in pkmus[0]._metadata:
            try:
                kwargs[key] = numpy.mean([getattr(pkmu,key) for pkmu in pkmus])
            except:
                kwargs[key] = getattr(pkmus[0],key)
            
        
        # add the named keywords, if present
        kwargs['force_index_match'] = getattr(pkmus[0], 'force_index_match', False)
        kwargs['sum_only'] = getattr(pkmus[0], 'sum_only', [])
        
        return PkmuResult(pkmus[0].kedges, pkmus[0].muedges, data, **kwargs)
        
    #--------------------------------------------------------------------------
    # convenience properties
    #--------------------------------------------------------------------------
    @property
    def columns(self):
        return list(self.data.dtype.names)
        
    @property
    def k_center(self):
        return self.index['k_center'][:,0]
    
    @property
    def mu_center(self):
        return self.index['mu_center'][0,:]
    
    @property
    def Nmu(self):
        return self.data.shape[1]
        
    @property
    def Nk(self):
        return self.data.shape[0]
        
    #--------------------------------------------------------------------------
    # utility functions
    #--------------------------------------------------------------------------
    def _get_index(self, name, val):

        index = self.k_center
        if name == 'mu' or name == 1:
            index = self.mu_center

        if self.force_index_match:
            i = (numpy.abs(index-val)).argmin()
        else:
            try:
                i = list(index).index(val)
            except Exception as e:
                raise IndexError("error converting %s index; try setting " %name + 
                                 "`force_index_match=True`: %s" %str(e))
                
        return i
        
    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def add_column(self, name, data):
        """
        Add a column with the name ``name`` to the data stored in ``self.data`
        
        Notes
        -----
        A new mask is calculated, with any elements in the new data masked
        if they are not finite.
        
        Parameters
        ----------
        name : str
            the name of the new data to be added to the structured array
        data : numpy.ndarray
            a numpy array to be added to ``self.data``, which must be the same
            shape as ``self.data``
        """
        if numpy.shape(data) != self.data.shape:
            raise ValueError("data to be added must have shape %s" %str(self.data.shape))
            
        dtype = self.data.dtype.descr
        if name not in self.data.dtype.names:
            dtype += [(name, data.dtype.type)]
            
        new = numpy.zeros(self.data.shape, dtype=dtype)
        mask = numpy.zeros(self.data.shape, dtype=bool)
        for col in self.columns:
            new[col] = self.data[col]
            mask = numpy.logical_or(mask, ~numpy.isfinite(new[col]))
            
        new[name] = data
        mask = numpy.logical_or(mask, ~numpy.isfinite(new[name]))
        
        self.data = numpy.ma.array(new, mask=mask)
            
    def nearest_bin_center(self, name, val):
        """
        Return the nearest `k` or `mu` bin center value to the value `val`
        
        Parameters
        ----------
        name : int or string
            If an int is passed, must be `0` for `k` or `1` for `mu`. If 
            a string is passed, must be either `k` or `mu` 
        val : float
            The `k` or `mu` value that we want to find the nearest bin to
            
        Returns
        -------
        index_val : float
            The center value of the bin closest to `val`
        """
        # verify input
        if isinstance(name, basestring):
            if name not in ['k','mu']:
                raise ValueError("`name` argument must be `k` or `mu`, if string")
        elif isinstance(name, int):
            if name not in [0, 1]:
                raise ValueError("`name` argument must be 0 for `k` or 1 for `mu`, if int")
        else:
            raise ValueError("`name` argument must be an int or string")
        
        index = self.k_center
        if name == 'mu' or name == 1:
            index = self.mu_center

        i = (numpy.abs(index-val)).argmin()
        return index[i]
        
    def Pk(self, mu=None, weights=None):
        """
        Return the power measured P(k) at a specific value of mu, as a 
        masked numpy recarray. If no `mu` is provided, return the
        data averaged over all mu bins, optionally weighted by
        `weights`
        
        Notes
        -----
        *   `mu` can be either an integer specifying which bin, or the
            center value of the bin itself. 
        *   If `mu` gives the bin value and `force_index_match` is 
            False, then the value must be present in `mu_center`. If 
            `force_index_match` is True, then it returns the nearest 
            bin to the value specified
        
        Parameters
        ---------
        mu : int or float
            The mu bin to select. If a `float`, `mu` must be a value 
            in `self.mu_center`.
            
        Returns
        -------
        Pk : numpy.ma.core.MaskedArray
            A masked structured array specifying the P(k) slice at the 
            mu-bin specified
        """
        # return the average over mu
        if mu is None:
            index = [self.index['k_center'], self.index['mu_center']]
            edges = [self.kedges, self.muedges]
            Pk = self._reindex(1, index, edges, numpy.linspace(edges[1][0],edges[1][-1],2), weights)
            return numpy.squeeze(Pk.data)
        # return a specific mu slice
        else:
            if not isinstance(mu, int): 
                mu = self._get_index('mu', mu)
        
            return self.data[:,mu]
        
    def Pmu(self, k=None, weights=None):
        """
        Return the power measured P(mu) at a specific value of k, as a 
        masked numpy recarray. If no `k` is provided, return the
        data averaged over all k bins, optionally weighted by
        `weights`
        
        Notes
        -----
        *   `k` can be either an integer specifying which bin, or the
            center value of the bin itself. 
        *   If `k` gives the bin value and `force_index_match` is 
            False, then the value must be present in `k_center`. If 
            `force_index_match` is True, then it returns the nearest 
            bin to the value specified
        
        Parameters
        ---------
        k : int or float
            The k bin to select. If a `float`, `k` must be a value 
            in `self.k_center`.
            
        Returns
        -------
        Pmu : numpy.ma.core.MaskedArray
            A masked structured array specifying the P(mu) slice at the 
            k-bin specified
        """
        # return the average over k
        if k is None:
            index = [self.index['k_center'], self.index['mu_center']]
            edges = [self.kedges, self.muedges]
            Pk = self._reindex(0, index, edges, numpy.linspace(edges[0][0],edges[0][-1],2), weights)
            return numpy.squeeze(Pk.data)
        # return a specific k slice
        else:            
            if not isinstance(k, int): 
                k = self._get_index('k', k)
        
            return self.data[k,:]
    
    def reindex_k(self, dk, weights=None, force=True, return_spacing=False):
        """
        Reindex the k dimension and return a PkmuResult holding
        the re-binned data, optionally weighted by `weights`. 
        
        Notes
        -----
        We can only re-bin to an integral factor of the current 
        dimension size in order to inaccuracies when re-binning to 
        overlapping bins
        
        
        Parameters
        ----------
        dk : float
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
        pkmu : PkmuResult
            class holding the re-binned results
        dk : float, optional
            the spacing used in the re-binned data
        """        
        # determine the new binning
        old_dk = numpy.diff(self.k_center)[0]
        factor = numpy.round(dk/old_dk).astype('int')
        if not factor:
            raise ValueError("new spacing must be smaller than original spacing of %.2e h/Mpc" %old_dk)
        if factor == 1:
            raise ValueError("closest binning size to input `dk` is the same as current binning")
        if old_dk*factor != dk and not force: 
            raise ValueError("if `force = False`, new bin spacing must be an integral factor smaller than original")
        
        # make a copy of the data
        data = self.data.copy()
                
        # get the weights
        if isinstance(weights, basestring):
            if weights not in self.columns:
                raise ValueError("Cannot weight by `%s`; no such column" %weights)
            weights = self.data[weights].data
            
        # check if we need to discard bins from the end
        leftover = self.Nk % factor
        if leftover and not force:
            args = (leftover, old_dk*factor)
            raise ValueError("cannot re-bin because they are %d extra bins, using dk = %.2e h/Mpc" %args)
        if leftover:
            data = data[:-leftover,:]
            if weights is not None: weights = weights[:-leftover, :]
        
        # new edges
        new_shape = (self.Nk/factor, self.Nmu)
        new_kedges = numpy.linspace(self.kedges[0], self.kedges[-1], new_shape[0]+1)
        
        # the re-binned data
        new_data = {}
        for col in self.columns:
            operation = numpy.nanmean
            if weights is not None or col in self.sum_only:
                operation = numpy.nansum
            new_data[col] = bin_ndarray(data[col].data, new_shape, weights=weights, operation=operation)
            
        meta = {k:getattr(self,k) for k in self._metadata}
        pkmu = PkmuResult(new_kedges, self.muedges, new_data, self.force_index_match, self.sum_only, **meta)
        return (pkmu, dk) if return_spacing else pkmu
