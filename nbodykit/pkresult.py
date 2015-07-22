from pkmuresult import rebin
import numpy

class PkResult(object):
    """
    PkResult provides an interface to store and manipulate a 1D power 
    spectrum measurement, as a function of wavenumber `k`.
    
    Notes
    -----
    The `data` attribute can be accessed using the __getitem__ behavior
    of `PkResult`. Additionally, __getitem__ can be passed `k` values, 
    which can refer to the bin center values. If 
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
    columns : list of str
        a list of the names of the fields in `data`
    kedges  : array_like
        the edges of the k bins used. shape is `Nk+1`
    force_index_match : bool
        If `True`, when indexing using `k` or `mu` values, return
        results for the nearest bin to the value specified
    sum_only : list of str
        A list of strings specifying fields in `data` that will 
        only be summed when combining bins.
    """
    def __init__(self, kedges, data, force_index_match=False, 
                  sum_only=[], **kwargs):
        """
        Parameters
        ----------
        kedges : array_like
            The list of the edges of the k bins
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
        # name of the columns
        columns = data.keys()
        self.kedges = kedges
        
        # treat any NaNs as missing data
        mask = numpy.zeros(len(kedges)-1, dtype=bool)
        dtypes = []
        for name in columns:
            mask = numpy.logical_or(mask, ~numpy.isfinite(data[name]))
            dtypes.append((name, data[name].dtype))

        # create a structured array to store the data
        shape = (len(self.kedges)-1)
        self.data = numpy.empty(shape, dtype=numpy.dtype(dtypes))
        for name in columns:
            self.data[name] = data[name]
        
        # now make it a masked array
        self.data = numpy.ma.array(self.data, mask=mask)
        
        # match closest index always returns nearest bin value
        self.force_index_match = force_index_match
        
        # fields which are averaged
        self.sum_only = sum_only
        
        # save any metadata too
        self._metadata = []
        for k, v in kwargs.iteritems():
            self._metadata.append(k)
            setattr(self, k, v)
    
    def __getitem__(self, key):
        if key in self.columns:
            return self.data[key]
            
        try:
            if isinstance(key, slice):
                new_slice = []
                for name in ['start', 'stop']:
                    val = getattr(key, name)
                    if not isinstance(val, int) and val is not None:
                        new_slice.append(self._get_index(val))
                    else:
                        new_slice.append(val)
                new_key = slice(*new_slice)
            elif not isinstance(key, int):
                new_key = self._get_index(key)
            else:
                new_key = key       
            return self.data[new_key]
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
        return PkResult(d['kedges'], data, **kwargs)
        
    @classmethod
    def from_dict(cls, d, names, **kwargs):
        """
        Return a PkmuResult object from a data array and column names. 
        Additional metadata can be specified as keyword arguments
        
        Notes
        -----
        `edges` must be passed as a keyword
        
        Parameters
        ----------
        d : dict
            dictionary of arrays to be stored as the data columns and bin 
            edges
        **kwargs 
            any additional keywords or metadata to be stored
        """
        if 'edges' not in kwargs:
            raise ValueError("must supply `edges` data as keyword")
        if d.shape[-1] != len(names):
            raise ValueError("a name for each of the %d data columns must be provided" %d.shape[-1])
        edges = kwargs.pop('edges')
        data_dict = {col:d[:,i] for i,col in enumerate(names)}
        return PkResult(edges, data_dict, **kwargs)
        
    @classmethod
    def from_list(cls, pks, weights=None, sum_only=[]):
        """
        Return an average PkResult object from a list of PkResult 
        objects, optionally using weights. For those columns in 
        sum_only, the data will be summed and not averaged.
        
        Notes
        -----
        The input power spectra must be defined on the same grid
        and have the same data columns.
        
        Parameters
        ----------
        pkmus : list of PkmuResult
            the list of 1D power spectrum objects to average
        
        """
        # check columns for all objects
        columns = [pk.columns for pk in pks]
        if not all(sorted(cols) == sorted(columns[0]) for cols in columns):
            raise ValueError("cannot combine PkResults with different column names")
        
        # check edges too
        edges = [getattr(pk,'kedges') for pk in pks]
        if not all(numpy.allclose(e, edges[0]) for e in edges):
            raise ValueError("cannot combine PkResults with different kedges")
        
        # compute the weights
        if weights is None:
            weights = numpy.ones((len(pks), pks[0].Nk))
        else:
            if isinstance(weights, basestring):
                if weights not in columns[0]:
                    raise ValueError("Cannot weight by `%s`; no such column" %weights)
                weights = numpy.array([pk.data[weights].data for pk in pks])
        
        # take the mean or the sum
        data = {}    
        for name in columns[0]:
            col_data = numpy.array([pk.data[name] for pk in pks])
            if name not in sum_only:
                with numpy.errstate(invalid='ignore'):
                    data[name] = (col_data*weights).sum(axis=0) / weights.sum(axis=0)
            else:
                data[name] = numpy.sum(col_data, axis=0)
            
        # handle the metadata
        kwargs = {}
        for key in pks[0]._metadata:
            try:
                kwargs[key] = numpy.mean([getattr(pk,key) for pk in pks])
            except:
                kwargs[key] = getattr(pks[0],key)
            
        
        # add the named keywords, if present
        kwargs['force_index_match'] = getattr(pks[0], 'force_index_match', False)
        kwargs['sum_only'] = getattr(pks[0], 'sum_only', [])
        
        return PkResult(pks[0].kedges, data, **kwargs)
        
    #--------------------------------------------------------------------------
    # convenience properties
    #--------------------------------------------------------------------------
    @property
    def columns(self):
        return list(self.data.dtype.names)
        
    @property
    def k_center(self):
        return 0.5*(self.kedges[1:]+self.kedges[:-1])
            
    @property
    def Nk(self):
        return self.data.shape[0]
        
    #--------------------------------------------------------------------------
    # utility functions
    #--------------------------------------------------------------------------
    def _get_index(self, val):

        index = self.k_center
        if self.force_index_match:
            i = (numpy.abs(index-val)).argmin()
        else:
            try:
                i = list(index).index(val)
            except Exception as e:
                raise IndexError("error converting k index; try setting " + 
                                 "`force_index_match=True`: %s" %str(e))
                
        return i
            
    def _reindex(self, index, edges, bins, weights):
        
        # compute the bins
        N_old = index.shape
        if isinstance(bins, int):
            if bins >= N_old:
                raise ValueError("Can only reindex into fewer than %d bins" %N_old)
            bins = numpy.linspace(edges[0], edges[-1], bins+1)
        else:
            if len(bins) >= N_old:
                raise ValueError("Can only reindex into fewer than %d bins" %N_old)
        
        # compute the weights
        if weights is None:
            weights = numpy.ones(self.Nk)
        else:
            if isinstance(weights, basestring):
                if weights not in self.columns:
                    raise ValueError("Cannot weight by `%s`; no such column" %weights)
                weights = self.data[weights].data
            
        # get the rebinned data
        edges = bins
        new_data = rebin(index, edges, self.data, weights, self.sum_only)
        
        # return a new PkResult
        meta = {k:getattr(self,k) for k in self._metadata}
        return PkResult(edges, new_data, self.force_index_match, self.sum_only, **meta)
        
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
        
    def nearest_bin_center(self, val):
        """
        Return the nearest `k` bin center value to the value `val`
        
        Parameters
        ----------
        val : float
            The `k` value that we want to find the nearest bin to
            
        Returns
        -------
        index_val : float
            The center value of the bin closest to `val`
        """
        index = self.k_center
        i = (numpy.abs(index-val)).argmin()
        return index[i]
    
    def reindex_k(self, bins, weights=None):
        """
        Reindex the k dimension and return a PkResult holding
        the re-binned data, optionally weighted by `weights`
        
        
        Parameters
        ---------
        bins : integer or array_like
            If an integer is given, `bins+1` edges are used. If a sequence,
            then the values should specify the bin edges.
        weights : str or array_like, optional
            If a string is given, it is intepreted as the name of a 
            data column in `self.data`. If a sequence is passed, then the
            shape must be equal to (`self.Nk`, `self.Nmu`)
            
        Returns
        -------
        pk : PkResult
            class holding the re-binned results
        """
        index = self.k_center
        return self._reindex(self.k_center, self.kedges, bins, weights)