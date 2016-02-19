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
import numpy
from nbodykit.utils.config import autoassign, ConstructorSchema, ReadConfigFile
from argparse import Namespace

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

def get_plugin_comm():
    """
    Return the global MPI communicator that all plugins 
    will be instantiated with (stored in `comm` attribute of plugin)
    """
    return _comm
    
def set_plugin_comm(comm):
    """
    Set the global MPI communicator that all plugins 
    will be instantiated with (stored in `comm` attribute of plugin)
    """
    global _comm
    _comm = comm

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
            cls.__init__.__func__.schema = ConstructorSchema()
            cls.schema = cls.__init__.schema

            # track names of classes
            setattr(cls.registry, cls.plugin_name, cls)

            # register the class
            cls.register()

            # attach the global communincator
            cls.comm = get_plugin_comm()
            
            # if a `DataSource`, inject the 'cosmo' keyword
            extra = []
            if issubclass(cls, DataSource):
                if 'cosmo' not in cls.schema:
                    h = 'the `Cosmology` class relevant for the DataSource'
                    cls.schema.add_argument("cosmo", default=None, help=h)
                    extra.append('cosmo')               
                
            # configure the class __init__
            cls.__init__ = autoassign(cls.__init__.__func__, allowed=extra)
            
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
                    cast = klass.schema[k].type
                    if cast is not None: 
                        kwargs[k] = cast(kwargs[k])
        return klass(**kwargs)

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
    
    readall: method
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

    def readall(self, columns):
        """ 
        Override to provide a method to read in all data at once,
        uncollectively. 

        Notes
        -----

        This function will be called by the default 'read' function
        on the root rank to read in the data set.
        The intention is to reduce the complexity of implementing a
        simple and small data source.
            
        """
        raise NotImplementedError

    def read(self, columns, stat, full=False):
        """
        Yield the data in the columns. If full is True, read all
        particles in one run; otherwise try to read in chunks.

        Override this function for complex, large data sets. The read
        operation shall be collective, each yield generates different
        sections of the datasource.

        On every iteration `stat` shall be updated with the global 
        statistics. Current keys are `Ntot`    
        """
        if self.comm.rank == 0:
            
            # make sure we have at least one column to read
            if not len(columns):
                raise RuntimeError("DataSource::read received no columns to read")
            
            data = self.readall(columns)    
            shape_and_dtype = [(d.shape, d.dtype) for d in data]
            Ntot = len(data[0]) # columns has to have length >= 1, or we crashed already
            
            # make sure the number of rows in each column read is equal
            if not all(len(d) == Ntot for d in data):
                raise RuntimeError("column length mismatch in DataSource::read")
        else:
            shape_and_dtype = None
            Ntot = None
        shape_and_dtype = self.comm.bcast(shape_and_dtype)
        stat['Ntot'] = self.comm.bcast(Ntot)

        if self.comm.rank != 0:
            data = [
                numpy.empty(0, dtype=(dtype, shape[1:]))
                for shape,dtype in shape_and_dtype
            ]

        yield data 


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
            Paint from a data source. It shall loop over self.read_and_decompose(...)
            and paint the data in chunks.
        """
        raise NotImplementedError

    def read_and_decompose(self, pm, datasource, columns, stats):

        assert 'Position' in columns
        assert pm.comm == self.comm # pm must be from the same communicator!

        for data in datasource.read(columns, stats, full=False):
            data = dict(zip(columns, data))
            position = data['Position']

            layout = pm.decompose(position)

            for c in columns:
                data[c] = layout.exchange(data[c])
                
            yield [data[c] for c in columns]

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
    def parse_known_yaml(kls, name, config_file):
        """
        Parse the known (and unknown) attributes from a YAML, where `known`
        arguments must be part of the Algorithm.parser instance
        """
        # get the class for this algorithm name
        klass = getattr(kls.registry, name)
        
        # get the namespace from the config file
        return ReadConfigFile(config_file, klass.schema)


__valid__ = [DataSource, Painter, Transfer, Algorithm]
__all__ = list(map(str, __valid__))

def isplugin(name):
    """
    Return `True`, if `name` is a registered plugin for any extension point
    """
    for extensionpt in __valid__:
        if name in extensionpt.registry: return True
    
    return False
    
def get_extensionpt(plugin_name):
    """
    Return `True`, if `name` is a registered plugin for any extension point
    """
    if not isplugin(plugin_name):
        raise ValueError("'%s' does not match the names of any loaded plugins" %plugin_name)
        
    for extensionpt in __valid__:
        if plugin_name in extensionpt.registry:
            return extensionpt
