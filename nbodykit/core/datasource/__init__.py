from ...plugins import PluginBase, PluginBaseMeta, MetaclassWithHooks
from ...plugins.hooks import attach_cosmo
from ...distributedarray import ScatterArray
from ...extern.six import add_metaclass

from abc import abstractmethod
import numpy
import weakref


class DataStream(object):
    """
    A class to represent an open :class:`DataSource` stream.
    
    The class behaves similar to the built-in :func:`open` for 
    :obj:`file` objects. The stream is returned by calling 
    :func:`DataSource.open` and the class is written such that it 
    can be used with or without the `with` statement, similar to 
    the file :func:`open` function.
    
    Attributes
    ----------
    read : callable
        a method that returns an iterator that will iterate through
        the data source (either collectively or uncollectively), 
        returning the specified columns
    nread : int
        during each iteration of `read`, this will store the number
        of rows read from the data source
    """
    def __init__ (self, data, defaults={}):
        """
        Parameters
        ----------
        data : DataSource
            the DataSource that this stream returns data from
        defaults : dict
            dictionary of default values for the specified columns
        """
        # store the data and defaults
        self.data = data
        self.defaults = defaults
        self.nread = 0
        
        # store a weak reference to the private cache
        self._cacheref = data._cache
        if not self._cacheref.empty:
            self.nread = self._cacheref.total_size
            
    def __enter__ (self):
        return self
        
    def __exit__ (self, exc_type, exc_value, traceback):
        self.close()
        
    def close(self):
        """
        Close the DataStream; no `read` operations are
        allowed on closed streams
        """
        if not self.closed:
            del self._cacheref
        
    @property
    def closed(self):
        """
        Return whether or not the DataStream is closed
        """
        return not hasattr(self, '_cacheref')        
        
    def isdefault(self, name, data):
        """
        Return `True` if the input data is equal to the default
        value for this stream
        """
        if name in self.defaults:
            return self.defaults[name] == data[0]
        else:
            return False
        
    def read(self, columns, full=False):
        """
        Read the data corresponding to `columns`
        
        Parameters
        ----------
        columns : list
            list of strings giving the names of the desired columns
        full : bool, optional
            if `True`, any `bunchsize` parameters will be ignored, so 
            that each rank will read all of its specified data section
            at once; default is `False`
        
        Returns
        -------
        data : list
            a list of the data for each column in `columns`
        """
        if self.closed:
            raise ValueError("'read' operation on closed data stream")
            
        # return data from cache, if it's not empty
        if not self._cacheref.empty:
            
            # valid column names
            valid_columns = list(set(self._cacheref.columns)|set(self.defaults))
            
            # replace any missing columns with None
            data = []
            for i, col in enumerate(columns):
                if col in self._cacheref.columns:
                    data.append(self._cacheref[col])
                else:
                    data.append(None)
            
            # yield the blended data
            yield self._blend_data(columns, data, valid_columns, self._cacheref.local_size)
            
        # parallel read is defined
        else:
            # reset the number of rows read
            self.nread = 0
            
            # do the parallel read
            for data in self.data.parallel_read(columns, full=full):
                
                # determine the valid columns (where data is not None)
                valid_columns = []; indices = []
                for i, col in enumerate(columns):
                    if data[i] is not None:
                        valid_columns.append(col)
                        indices.append(i)
                valid_columns = list(set(valid_columns)|set(self.defaults))
                
                # verify data
                valid_data = [data[i] for i in indices]
                size = self.data._verify_data(valid_data)
                
                # update the number of rows read
                self.nread += self.data.comm.allreduce(size)
                
                # yield the blended data with defaults
                yield self._blend_data(columns, data, valid_columns, size)
                
    def _blend_data(self, columns, data, valid_columns, size):
        """
        Internal function to blend data that has been explicitly read
        and any default values that have been set. 
        
        Notes
        -----
        *   Default values are returned with the correct size but minimal
            memory usage using `stride_tricks` from `numpy.lib`
        *   This function will crash if a column is requested that the 
            data source does not provided and no default exists
        
        Parameters
        ----------
        columns : list
            list of the column names that are being returned
        data : list
            list of data corresponding to `columns`; if a column is not 
            supported, the element of data is `None`
        valid_columns : list
            the list of valid column names; the union of the columns supported
            by the data source and the default values
        size : int
            the size of the data we are returning
        
        Returns
        -------
        newdata : list
            the list of data with defaults blended in, if need be
        """
        newdata = []
        for i, col in enumerate(columns):
                        
            # this column is missing -- crash
            if col not in valid_columns:
                args = (col, str(valid_columns))
                raise DataSource.MissingColumn("column '%s' is unavailable; valid columns are: %s" %args)
            
            # we have this column
            else:
                # return read data, if not None
                if data[i] is not None:
                    newdata.append(data[i])
                # return the default
                else:
                    if col not in self.defaults:
                        raise RuntimeError("missing default value when trying to blend data")    
                        
                    # use stride_tricks to avoid memory usage
                    val = numpy.asarray(self.defaults[col])
                    d = numpy.lib.stride_tricks.as_strided(val, (size, val.size), (0, val.itemsize))
                    newdata.append(numpy.squeeze(d))
            
        return newdata
        
class DataCache(object):
    """
    A class to cache data in a manner that can be weakly
    referenced via `weakref`
    """
    def __init__(self, columns, data, local_size, total_size):
        """
        Parameters
        ----------
        columns : list of str
            the list of columns in the cache
        data : list of array_like
            a list of data for each column
        local_size : int
            the size of the data stored locally on this rank
        total_size : int
            the global size of the data
        """
        self.columns    = columns
        self.local_size = local_size
        self.total_size = total_size
    
        for col, d in zip(columns, data):
            setattr(self, col, d)
    
    @property
    def empty(self):
        """
        Return whether or not the cache is empty
        """
        return len(self.columns) == 0
        
    def __getitem__(self, col):
        if col in self.columns:
            return getattr(self, col)
        else:
            raise KeyError("no such column in DataCache; available columns: %s" %str(self.columns))    
    
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return 'DataCache(%s)' %str(self.columns)
        

# attach the cosmology to data sources
DataSourceMeta = MetaclassWithHooks(PluginBaseMeta, attach_cosmo)

@add_metaclass(DataSourceMeta)
class DataSourceBase(PluginBase):

    @staticmethod
    def BoxSizeParser(value):
        """
        Read the `BoxSize`, enforcing that the BoxSize must be a 
        scalar or 3-vector

        Returns
        -------
        BoxSize : array_like
            an array of size 3 giving the box size in each dimension
        """
        boxsize = numpy.empty(3, dtype='f8')
        try:
            if isinstance(value, (tuple, list)) and len(value) != 3:
                raise ValueError
            boxsize[:] = value
        except:
            raise ValueError("DataSource `BoxSize` must be a scalar or three-vector")
        return boxsize

    pass

class GridSource(DataSourceBase):
    """
    A DataSource reading directly already on a grid
    """
    @abstractmethod
    def read(self, real):
        """
        Read into a real field
        """
        pass

class DataSource(DataSourceBase):
    """
    Mount point for plugins which refer to the reading of input files.
    The `read` operation occurs on a :class:`DataStream` object, which
    is returned by :func:`~DataSource.open`.
    
    Default values for any columns to read can be supplied as a 
    dictionary argument to :func:`~DataSource.open`.
    
    Plugins of this type should provide the following attributes:

    plugin_name : str
        A class attribute that defines the name of the plugin in 
        the registry
    
    register : classmethod
        A class method taking no arguments that updates the
        :class:`~nbodykit.utils.config.ConstructorSchema` with
        the arguments needed to initialize the class
    
    readall: method
        A method to read all available data at once (uncollectively) 
        and cache the data in memory for repeated calls to `read`

    parallel_read: method
        A method to read data for complex, large data sets. The read
        operation shall be collective, with each yield generating 
        different sections of the data source on different ranks. 
        No caching of data takes places.
    
    Notes
    -----
    *   a :class:`~nbodykit.cosmology.Cosmology` instance can be passed to 
        any DataSource class via the `cosmo` keyword
    *   the data will be cached in memory if returned via :func:`~DataStream.readall`
    *   the default cache behavior is for the cache to persist while 
        an open DataStream remains, but the cache can be forced
        to persist via the :func:`DataSource.keep_cache` context manager
    """
    class MissingColumn(Exception):
        pass
            
    def _cache_data(self):
        """
        Internal function to cache the data from `readall`.
        
        This function performs the following actions:
        
             1. calls `readall` on the root rank
             2. scatters read data evenly across all ranks
             3. stores the scattered data as the `cache` attribute
             4. stores the total size of the data as the `size` attribute
             5. stores the available columns in the cache as `cached_columns`
             
        Returns
        -------
        success : bool
            if the call to `readall` is successful, returns `True`, else `False`
        """
        # rank 0 tries to readall, and tells everyone else if it succeeds
        success = False
        if self.comm.rank == 0:
            
            # read all available data
            try:
                data = self.readall()
                success = True
            except NotImplementedError:
                pass
            except Exception:
                import traceback
                raise Exception("an unknown exception occurred while trying to cache data via `readall`:\n"
                                "traceback:\n\n%s" %traceback.format_exc())
                
        # determine if readall was successful
        success = self.comm.bcast(success, root=0)
    
        # cache the data
        if success:
            
            columns = []; size = None
            if self.comm.rank == 0:
                columns = list(data.keys())
                data = [data[c] for c in columns] # store a list
            
            # everyone gets the column names
            columns = self.comm.bcast(columns, root=0)
        
            # verify the input data
            if self.comm.rank == 0:    
                size = self._verify_data(data)
            else:
                data = [None for c in columns]
        
            # everyone gets the total size
            size = self.comm.bcast(size, root=0)
        
            # scatter the data across all ranks
            # each rank caches only a part of the total data
            cache = []; local_sizes = []
            for d in data:
                cache.append(ScatterArray(d, self.comm, root=0))
                local_sizes.append(len(cache[-1]))
                
            # this should hopefully never fail (guaranted to have nonzero length)
            if not all(s == local_sizes[0] for s in local_sizes):
                raise RuntimeError("scattering data resulted in uneven lengths between columns")
            local_size = local_sizes[0]

            # the total collective size of the datasource
            self.size = size # this will persist, even if cache is deleted
            
            # return the cache
            return DataCache(columns, cache, local_size, size)
        else:
            return DataCache([], [], 0, 0) # empty cache

    def _verify_data(self, data):
        """
        Internal function to verify the input data by checking that: 
            
            1. `data` is not empty
            2. the sizes of each element of data are equal
        
        Parameters
        ----------
        data : list
            the list of data arrays corresponding to the requested columns
        
        Returns
        -------
        size : int
            the size of each element of `data`
        """
        # need some data
        if not len(data):
            raise ValueError('DataSource::read did not return any data')
        
        # make sure the number of rows in each column read is equal
        # columns has to have length >= 1, or we crashed already
        if not all(len(d) == len(data[0]) for d in data):
            raise RuntimeError("column length mismatch in DataSource::read")
        
        # return the size
        return len(data[0])
                    
    @property
    def _cache(self):
        """
        Internal cache property storing a `DataCache`. This is designed to 
        be accessed only by `DataStream` objects
        """
        # create the weakref dict if need be
        if not hasattr(self, '_weakcache'):
            self._weakcache = weakref.WeakValueDictionary()
            
        # create the DataCache if need be
        if 'cache' not in self._weakcache:
            cache = self._cache_data()
            self._weakcache['cache'] = cache
            
        return self._weakcache['cache']
    
    @property
    def size(self):
        """
        The total size of the DataSource.
        
        The user can set this explicitly (only once per datasource) 
        if the size is known before :func:`DataStream.read` is called
        """
        try:
            return self._size
        except:
            raise AttributeError("DataSource `size` is not known a priori (i.e., before the 'read' operation)")
    
    @size.setter
    def size(self, val):
        """
        Set the `size` attribute. This should only be set once
        """
        if not hasattr(self, '_size'):
            self._size = val
        else:
            if val != self._size:
                raise ValueError("DataSource `size` has already been set to a different value")
    
    def keep_cache(self):
        """
        A context manager that forces the DataSource cache to persist,
        even if there are no open DataStream objects. This will 
        prevent unwanted and unnecessary re-readings of the DataSource.

        The below example details the intended usage. In this example,
        the data is cached only once, and no re-reading of the data
        occurs when the second stream is opened.
        
        .. code-block:: python
        
            with datasource.keep_cache():
                with datasource.open() as stream1:
                    [[pos]] = stream1.read(['Position'], full=True)
                
                with datasource.open() as stream2:
                    [[vel]] = stream2.read(['Velocity'], full=True)
        """
        # simplest implementation is returning a stream
        return DataStream(self) 
    
    def open(self, defaults={}):
        """
        Open the DataSource by returning a DataStream from which
        the data can be read. 
        
        This function also specifies the default values for any columns
        that are not supported by the DataSource. The defaults are
        unique to each DataStream, but a DataSource can be opened
        multiple times (returning different streams) with different 
        default values
        
        Parameters
        ----------
        defaults : dict, optional
            a dictionary providing default values for a given column
        
        Returns
        -------
        stream : DataStream
            the stream object from which the data can be read via
            :func:`~DataStream.read` function
        """
        # note: if DataSource is already `open`, we can 
        # still get a new stream with different defaults
        return DataStream(self, defaults=defaults)
                             
    def readall(self):
        """ 
        Override to provide a method to read all available data at once 
        (uncollectively) and cache the data in memory for repeated 
        calls to `read`

        Notes
        -----
        *   By default, :func:`DataStream.read` calls this function on the
            root rank to read all available data, and then scatters
            the data evenly across all available ranks
        *   The intention is to reduce the complexity of implementing a 
            simple and small data source, for which reading all data at once
            is feasible
            
        Returns
        -------
        data : dict
            a dictionary of all supported data for the data source; keys
            give the column names and values are numpy arrays
        """
        raise NotImplementedError
        
    def parallel_read(self, columns, full=False):
        """ 
        Override this function for complex, large data sets. The read
        operation shall be collective, each yield generates different
        sections of the datasource. No caching of data takes places.
        
        If the DataSource does not provide a column in `columns`, 
        `None` should be returned.
        
        Notes
        -----
        *   This function will be called if :func:`DataStream.readall` is 
            not implemented
        *   The intention is for this function to handle complex and 
            large data sets, where parallel I/O across ranks is 
            required to avoid memory and I/O issues
            
        Parameters
        ----------
        columns : list of str
            the list of data columns to return
        full : bool, optional
            if `True`, any `bunchsize` parameters will be ignored, so 
            that each rank will read all of its specified data section
            at once
        
        Returns
        -------
        data : list
            a list of the data for each column in columns; if the data source
            does not provide a given column, that element should be `None`
        """
        raise NotImplementedError
