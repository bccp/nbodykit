"""
    Declare `PluginMount` and various extension points

    To define a Plugin: 

    1.  subclass from the desired extension point class
    2.  define a class method `register` that fills the declares
        the relevant attributes by calling `add_argument`
        of `cls.schema`
    3. define a `plugin_name` class attribute

    To define an ExtensionPoint:

    1. add a class decorator @ExtensionPoint
"""
from nbodykit.utils.config import autoassign, ConstructorSchema, ReadConfigFile
from nbodykit.distributedarray import ScatterArray

import numpy
from argparse import Namespace
import functools

# MPI will be required because
# a plugin instance will be created for a MPI communicator.
from mpi4py import MPI

algorithms  = Namespace()
datasources = Namespace()
painters    = Namespace()
transfers   = Namespace()

# private variable to store global MPI communicator 
# that all plugins are initialized with
_comm = MPI.COMM_WORLD
_cosmo = None

def get_nbkit_comm():
    """
    Return the global MPI communicator that all plugins 
    will be instantiated with (stored in `comm` attribute of plugin)
    """
    return _comm
    
def set_nbkit_comm(comm):
    """
    Set the global MPI communicator that all plugins 
    will be instantiated with (stored in `comm` attribute of plugin)
    """
    global _comm
    _comm = comm
    
def get_nbkit_cosmo():
    """
    Return the global Cosmology instance
    """
    return _cosmo
    
def set_nbkit_cosmo(cosmo):
    """
    Set the global Cosmology instance
    """
    global _cosmo
    _cosmo = cosmo

class PluginInterface(object):
    """ 
    The basic interface of a plugin -- classes must implement
    the 'register' function
    """
    @classmethod
    def register(kls):
        raise NotImplementedError

        
def ExtensionPoint(registry):
    """ 
    Declares a class as an extension point, registering to registry 
    """
    def wrapped(cls):
        cls = add_metaclass(PluginMount)(cls)
        cls.registry = registry
        return cls
    return wrapped

class PluginMount(type):
    """ 
    Metaclass for an extension point that provides the 
    methods to manage plugins attached to the extension point
    """
    def __new__(cls, name, bases, attrs):
        # Add PluginInterface to the ExtensionPoint,
        # Plugins at an extensioni point will inherit from PluginInterface
        # This is more twisted than it could have been!

        if len(bases) == 0 or (len(bases) == 1 and bases[0] is object):
            bases = (PluginInterface,)
        return type.__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        # only executes when processing the mount point itself.
        # the extension mount point only declares a PluginInterface
        # the plugins at an extension point will always be its subclass
        
        if cls.__bases__ == (PluginInterface, ):
            return

        if not hasattr(cls, 'plugin_name'):
            raise RuntimeError("Plugin class must carry a 'plugin_name'")
        
        # register, if this plugin isn't yet
        if cls.plugin_name not in cls.registry:

            # initialize the schema and alias it
            if cls.__init__ == object.__init__:
                raise ValueError("please define an __init__ method for '%s'" %cls.__name__)
            
            # in python 2, __func__ needed to attach attributes to the real function; 
            # __func__ removed in python 3, so just attach to the function
            init = cls.__init__
            if hasattr(init, '__func__'):
                init = init.__func__
            
            # add a schema
            init.schema = ConstructorSchema()
            cls.schema = cls.__init__.schema

            # track names of classes
            setattr(cls.registry, cls.plugin_name, cls)

            # register the class
            cls.register()
            
            # configure the class __init__, attaching the comm, and optionally cosmo
            attach_cosmo = issubclass(cls, DataSource)
            cls.__init__ = autoassign(init, attach_cosmo=attach_cosmo)
            
    def create(cls, plugin_name, use_schema=False, **kwargs): 
        """ 
        Instantiate a plugin from this extension point,
        based on the name/value pairs passed as keywords. 
        
        Optionally, cast the keywords values, using the types
        defined by the schema of the class we are creating

        Parameters
        ----------
        plugin_name: str
            the name of the plugin to instantiate
        use_schema : bool, optional
            if `True`, cast the kwargs that are defined in 
            the class schema before initializing. Default: `False`
        kwargs : (key, value) pairs
            the parameter names and values that will be
            passed to the plugin's __init__

        Returns
        -------
        plugin : 
            the initialized instance of `plugin_name`
        """
        if plugin_name not in cls.registry:
            raise ValueError("'%s' does not match the names of any loaded plugins" %plugin_name)
            
        klass = getattr(cls.registry, plugin_name)
        
        # cast the input values, using the class schema
        if use_schema:
            for k in kwargs:
                if k in klass.schema:
                    arg = klass.schema[k]
                    kwargs[k] = klass.schema.cast(arg, kwargs[k])
                        
        toret = klass(**kwargs)
        
        ### FIXME: not always using create!!
        toret.string = id(toret)
        return toret
        
    def from_config(cls, parsed): 
        """ 
        Instantiate a plugin from this extension point,
        based on the input `parsed` value, which is parsed
        directly from the YAML configuration file
        
        There are several valid input cases for `parsed`:
            1.  parsed: dict
                containing the key `plugin`, which gives the name of 
                the Plugin to load; the rest of the dictionary is 
                treated as arguments of the Plugin
            2.  parsed: dict
                having only one entry, with key giving the Plugin name
                and value being a dictionary of arguments of the Plugin
            3.  parsed: dict
                if `from_config` is called directly from a Plugin class, 
                then `parsed` can be a dictionary of the named arguments,
                with the Plugin name inferred from the class `cls`
            4.  parsed: str
                the name of a Plugin, which will be created with 
                no arguments
        """    
        try:    
            if isinstance(parsed, dict):
                if 'plugin' in parsed:
                    kwargs = parsed.copy()
                    plugin_name = kwargs.pop('plugin')
                    return cls.create(plugin_name, use_schema=True, **kwargs)
                elif len(parsed) == 1:
                    k = list(parsed.keys())[0]
                    if isinstance(parsed[k], dict):
                        return cls.create(k, use_schema=True, **parsed[k])
                    else:
                        raise Exception
                elif hasattr(cls, 'plugin_name'):
                    return cls.create(cls.plugin_name, use_schema=True, **parsed)
                else:
                    raise Exception
            elif isinstance(parsed, str):
                return cls.create(parsed)
            else:
                raise Exception
        except Exception as e:
            import traceback
            traceback = traceback.format_exc()
            raise ValueError("failure to parse plugin:\n\n%s" %(traceback))
            
    def format_help(cls, *plugins):
        """
        Return a string specifying the `help` for each of the plugins
        specified, or all if none are specified
        """
        if not len(plugins):
            plugins = list(vars(cls.registry).keys())
            
        s = []
        for k in plugins:
            if not isplugin(k):
                raise ValueError("'%s' is not a valid Plugin name" %k)
            header = "Plugin : %s  ExtensionPoint : %s" % (k, cls.__name__)
            s.append(header)
            s.append("=" * (len(header)))
            s.append(getattr(cls.registry, k).schema.format_help())

        if not len(s):
            return "No available Plugins registered at %s" %cls.__name__
        else:
            return '\n'.join(s) + '\n'

# copied from six
def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper

@ExtensionPoint(transfers)
class Transfer:
    """
    Mount point for plugins which apply a k-space transfer function
    to the Fourier transfrom of a datasource field
    
    Plugins implementing this reference should provide the following 
    attributes:

    plugin_name : str
        class attribute giving the name of the subparser which 
        defines the necessary command line arguments for the plugin
    
    register : classmethod
        A class method taking no arguments that adds a subparser
        and the necessary command line arguments for the plugin
    
    __call__ : method
        function that will apply the transfer function to the complex array
    """
    def __call__(self, pm, complex):
        """ 
        Apply the transfer function to the complex field
        
        Parameters
        ----------
        pm : ParticleMesh
            the particle mesh object which holds possibly useful
            information, i.e, `w` or `k` arrays
        complex : array_like
            the complex array to apply the transfer to
        """
        raise NotImplementedError

class DataStream(object):
    """
    A class to represent an open `DataSource` stream, which
    is called by `DataSource.open()`
    
    The class is written such that it be used with or without
    the `with` statement, similar to the file `open()` function
    """
    def __init__ (self, data, defaults={}):
        self.data = data
        self.defaults = defaults
        
        # only cache if we aren't open yet
        if not self.isopen():
            if self.data._cache_data(): # fails if readall not defined
                self.data._has_readall = True
            # don't do anything if `readall` not defined    
            else:
                self.data._has_readall = False
                
        # store the defaults
        self.data.defaults = defaults
        
    def read(self, *args, **kwargs):
        return self.data.read(*args, **kwargs)
    
    def close(self):
        self.data.close()
    
    def isopen(self):
        return self.data.isopen()
        
    def __enter__ (self):
        return self
        
    def __exit__ (self, exc_type, exc_value, traceback):
        self.close()
        
            
@ExtensionPoint(datasources)
class DataSource:
    """
    Mount point for plugins which refer to the reading of input files 
    
    Notes
    -----
    *   a `Cosmology` instance can be passed to any `DataSource`
        class via the `cosmo` keyword

    Plugins implementing this reference should provide the following 
    attributes:

    plugin_name : str
        class attribute that defines the name of the Plugin in 
        the registry
    
    register : classmethod
        A class method taking no arguments that declares the
        relevant attributes for the class by adding them to the
        class schema
    
    simple_read: method
        A method that performs the reading of the field. This method
        reads in the full data set. It shall
        returns the position (in 0 to BoxSize) and velocity (in the
        same units as position). This method is called by the default
        read method on the root rank for reading small data sets.

    read: method
        A method that performs the reading of the field. It shall
        returns the position (in 0 to BoxSize) and velocity (in the
        same units as position), in chunks as an iterator. The
        default behavior is to use Rank 0 to read in the full data
        and yield an empty data. 
    """        
    @staticmethod
    def BoxSizeParser(value):
        """
        Read the `BoxSize, enforcing that the BoxSize must be a 
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
            raise ValueError("BoxSize must be a scalar or three-vector")
        return boxsize
            
    def open(self, defaults={}):
        """
        `Open` the DataSource
        """
        # note: if DataSource is already `open` (i.e., cached), we can 
        # still get a new stream with different defaults
        return DataStream(self, defaults=defaults)
            
    def isopen(self):
        """
        Return if the DataSource is open
        """
        return hasattr(self, '_has_readall')
    
    def close(self):
        """
        Close an open DataSource
        """
        # strict error catching
        if not self.isopen():
            raise ValueError("cannot perfom `close` operation on an already closed `DataSource`")
        
        # delete the cache and size if they exist
        if self._has_readall:
            if hasattr(self, 'cache'): del self.cache
            if hasattr(self, 'size'): del self.size
            
        # delete any defaults we defined
        if hasattr(self, 'defaults'):
            del self.defaults
            
        # this is created when 'open()' is called
        del self._has_readall
                 
    def read(self, columns, full=False):
        """
        Read the data corresponding to `columns` from an `open` DataSource
        """
        if not self.isopen():
            raise ValueError("cannot perfom `read` operation on a closed `DataSource`")
            
        # return data from cache
        if self._has_readall:
            
            # valid column names
            valid_columns = list(set(self.cached_columns)|set(self.defaults))
            
            # replace any missing columns with None
            data = []
            for i, col in enumerate(columns):
                if col in self.cached_columns:
                    index = self.cached_columns.index(col)
                    data.append(self.cache[index])
                else:
                    data.append(None)
            
            # yield the blended data
            yield self._blend_data(columns, data, valid_columns, self.size)
            
        # parallel read is defined
        else:
            
            for data in self.parallel_read(columns, full=full):
                
                # determine the valid columns (where data is not None)
                valid_columns = []; indices = []
                for i, col in enumerate(columns):
                    if data[i] is not None:
                        valid_columns.append(col)
                        indices.append(i)
                valid_columns = list(set(valid_columns)|set(self.defaults))
                
                # verify data
                valid_data = [data[i] for i in indices]
                size = self._verify_data(valid_data)
                
                # yield the blended data with defaults
                yield self._blend_data(columns, data, valid_columns, size)

    def readall(self):
        """ 
        Override to provide a method to read all available data at once 
        (uncollectively) and cache the data in memory for repeated 
        calls to `read`

        Notes
        -----
        *   The default 'read' function calls this function on the
            root rank to read all available data, and then scatters
            the data evenly across all available ranks
        *   The intention is to reduce the complexity of implementing a 
            simple and small data source, for which reading all data at once
            is feasible
            
        Returns
        -------
        data : dict
            a dictionary of all supported data for the datasource; keys
            give the column names and values are numpy arrays
        """
        raise NotImplementedError
        
    def parallel_read(self, columns, full=False):
        """ 
        Override this function for complex, large data sets. The read
        operation shall be collective, each yield generates different
        sections of the datasource. No caching of data takes places.
        
        If the `DataSource` does not provide a column in `columns`, 
        `None` should be returned
        
        Notes
        -----
        *   This function will be called if `readall` is not provided
        *   The intention is for this function to handle complex and 
            large data sets, where parallel I/O across ranks are 
            required to avoid memory and I/O issues
            
        Parameters
        ----------
        full : bool, optional
            if `True`, any `bunchsize` parameters will be ignored, so 
            that each rank will read all of its specified data section
            at once
        
        Returns
        -------
        data : list
            a list of the data for each column in columns; if the datasource
            does not provide a given column, that element should be `None`
        """
        raise NotImplementedError
        
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
        # rank 0 tries to read, and tells everyone else if it succeeds
        success = False
        if self.comm.rank == 0:
            
            # read all available data
            try:
                data = self.readall()
                success = True
            except:
                pass
                
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
            self.cache = []
            for d in data:
                self.cache.append(ScatterArray(d, self.comm, root=0))

            # store the size and column names
            self.size = size
            self.cached_columns = columns
            
            return True
        else:
            return False

    def _verify_data(self, data):
        """
        Internal function to verify the input data by checking that: 
            
            1. `data` is not empty
            2. the sizes of each element of data are equal
        
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
                raise ValueError("column '%s' is unavailable; valid columns are: %s" %args)
            
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
        

@ExtensionPoint(painters)
class Painter:
    """
    Mount point for plugins which refer to the painting of input files.

    Plugins implementing this reference should provide the following 
    attributes:

    plugin_name : str
        class attribute giving the name of the subparser which 
        defines the necessary command line arguments for the plugin
    
    register : classmethod
        A class method taking no arguments that adds a subparser
        and the necessary command line arguments for the plugin
    
    paint : method
        A method that performs the painting of the field.

    """
    
    def paint(self, pm, datasource):
        """ 
            Paint from a data source. 

            It shall read data then use basepaint to paint.
            
            Returns a dictionary of attributes of the painting operation.
        """
        raise NotImplementedError

    def basepaint(self, pm, position, weight=None):
        assert pm.comm == self.comm # pm must be from the same communicator!
        layout = pm.decompose(position)
        Nlocal = len(position)
        position = layout.exchange(position)
        if weight is not None:
            weight = layout.exchange(weight)
            pm.paint(position, weight)
        else:
            pm.paint(position) 
        return Nlocal 
        
@ExtensionPoint(algorithms)
class Algorithm:
    """
    Mount point for plugins which provide an interface for running
    one of the high-level algorithms, i.e, power spectrum calculation
    or FOF halo finder
    
    Plugins implementing this reference should provide the following 
    attributes:

    plugin_name : str
        class attribute giving the name of the subparser which 
        defines the necessary command line arguments for the plugin
    
    register : classmethod
        A class method taking no arguments that adds a subparser
        and the necessary command line arguments for the plugin
    
    __call__ : method
        function that will apply the transfer function to the complex array
    """        
    def run(self):
        raise NotImplementedError
    
    def save(self, *args, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def parse_known_yaml(kls, name, stream):
        """
        Parse the known (and unknown) attributes from a YAML, where `known`
        arguments must be part of the Algorithm.parser instance
        """
        # get the class for this algorithm name
        klass = getattr(kls.registry, name)
        
        # get the namespace from the config file
        return ReadConfigFile(stream, klass.schema)


__valid__ = [DataSource, Painter, Transfer, Algorithm]
__all__ = list(map(str, __valid__))

def isplugin(name):
    """
    Return `True`, if `name` is a registered plugin for any extension point
    """
    for extensionpt in __valid__:
        if name in extensionpt.registry: return True
    
    return False
