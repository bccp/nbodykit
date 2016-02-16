"""
    Declare PluginMount and various extension points.

    To define a Plugin, 

    1. subclass from the extension point class
    2. define a class method `register`, that calls add_argument
       to kls.parser.
    3. define a plugin_name member.

    To define an ExtensionPoint,

    1. add a class decorator @ExtensionPoint
    
"""

import numpy
# MPI will be required because
# a plugin instance will be created for a MPI communicator.
from mpi4py import MPI

from nbodykit.plugins import HelpFormatterColon, load
from argparse import Namespace, SUPPRESS, ArgumentParser, HelpFormatter

import sys
import contextlib

algorithms = Namespace()
datasources = Namespace()
painters = Namespace()
transfers = Namespace()
mstorages = Namespace()

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

def _parser_print_message(message, file=None):
    if file is None:
        import sys
        file = sys.stderr
    if _comm is not None and _comm.rank == 0:
        if message:
            file.write(message)

class PluginInterface(object):
    """ 
    The basic interface of a plugin 
    """
    @classmethod
    def register(kls):
        raise NotImplementedError

    def finalize_attributes(self):
        # override to finalize the attributes based on parsed attributes.
        pass

    def __eq__(self, other):
        return self.string == other.string

    def __ne__(self, other):
        return self.string != other.string

    def __init__(self, **kwargs):
    
        # automatically set the communicator for each plugin
        # to the global communicator, stored in _comm
        self.comm = _comm 

        # FIXME: set unique string by default
        # directly created object does not have a string!
        self.string = str(id(self)) 

        missing = []
        d = {}
       
        for action in self.parser._actions:
            argname = action.dest
            if action.default == SUPPRESS:
                continue
                    
            if argname not in kwargs:
                if not action.required:
                    d[argname] = action.default
                else:
                    missing += argname 
            else:
                d[argname] = kwargs[argname]
                kwargs.pop(argname)
                
        if len(missing):
            raise ValueError("Missing arguments : %s " % str(missing))
        if len(kwargs):
            raise ValueError("Extra arguments : %s " % str(list(kwargs.keys())))
        
        self.__dict__.update(d)
        
def ExtensionPoint(registry, help_formatter=HelpFormatter):
    """ Declares a class as an extension point, registering to registry """
    def wrapped(cls):
        cls = add_metaclass(PluginMount)(cls)
        cls.registry = registry
        cls.help_formatter = help_formatter
        cls.plugins = {}
        return cls
    return wrapped

class PluginMount(type):
    """ Metaclass for an extension point that provides
        the methods to manage
        plugins attached to the extension point.
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
            raise RuntimeError("Plugin class must carry a plugin_name.")
        
        # register, if this plugin isn't yet
        if cls.plugin_name not in cls.plugins:
            # add a commandline argument parser for each plugin
            # NOTE: we don't want every plugin to preparse sys.argv
            # so set args = ()
            cls.parser = ArgumentParser(cls.plugin_name, 
                                        usage=None, 
                                        add_help=False, 
                                        formatter_class=cls.help_formatter)

            cls.parser._print_message = _parser_print_message
            # track names of classes
            cls.plugins[cls.plugin_name] = cls
        
            # store as part of the algorithms namespace
            setattr(cls.registry, cls.plugin_name, cls)

            # try to call register class method
            if hasattr(cls, 'register'):
                cls.register()

            # set the class documentation automatically
            doc = cls.__doc__
            if doc is not None:
                cls.__doc__ += "\n\n"+cls.parser.format_help()
            else:
                cls.__doc__ = cls.parser.format_help()

    def create(kls, argv): 
        """ Instantiate a plugin from this extension point,
            based on the cmdline string. The arguments in string
            will be parsed and the attributes of the instance will
            be populated.

            Parameters
            ----------
            argv: list of strings
                The first field specifies the type of the plugin
                to create.
                The reset depends on the type of the plugin.

            Notes
            -----
            2nd stage parsing: A plugin can override 
            `finalize_attributes` to finalize the
            attribute values based on currently parsed attribute values.
        """
        klass = kls.plugins[argv[0]]
        ns = klass.parser.parse_args(argv[1:])
        
        self = klass(**vars(ns))
        self.string = str(argv)
        self.finalize_attributes()
        
        return self

    def format_help(kls):
        
        rt = []
        for k in kls.plugins:
            header = "Plugin : %s  ExtensionPoint : %s" % (k, kls.__name__)
            rt.append(header)
            rt.append("=" * (len(header)))
            rt.append(kls.plugins[k].parser.format_help())

        if not len(rt):
            return "No available Plugins registered at %s" % kls.__name__
        else:
            return '\n'.join(rt)

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

@ExtensionPoint(transfers, HelpFormatterColon)
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

    @classmethod
    def fromstring(kls, string): 
        return kls.create(string.split(':'))

@ExtensionPoint(datasources, HelpFormatterColon)
class DataSource:
    """
    Mount point for plugins which refer to the reading of input files 
    and the subsequent painting of those fields.

    Plugins implementing this reference should provide the following 
    attributes:

    plugin_name : str
        class attribute giving the name of the subparser which 
        defines the necessary command line arguments for the plugin
    
    register : classmethod
        A class method taking no arguments that adds a subparser
        and the necessary command line arguments for the plugin
    
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
        Parse a string of either a single float, or 
        a space-separated string of 3 floats, representing 
        a box size. Designed to be used by the DataSource plugins
        
        Returns
        -------
        BoxSize : array_like
            an array of size 3 giving the box size in each dimension
        """
        boxsize = numpy.empty(3, dtype='f8')
        sizes = [float(i) for i in value.split()]
        if len(sizes) == 1: sizes = sizes[0]
        boxsize[:] = sizes
        return boxsize

    def readall(self, columns):
        """ Override to provide a method to read in all data at once,
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
            statistics. Current keys are `Ntot`.
            
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

    @classmethod
    def fromstring(kls, string): 
        return kls.create(string.split(':'))



@ExtensionPoint(painters, HelpFormatterColon)
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

            for c in list(data.keys()):
                data[c] = layout.exchange(data[c])
                
            yield [data[c] for c in columns]

    @classmethod
    def fromstring(kls, string): 
        return kls.create(string.split(':'))

@ExtensionPoint(mstorages, HelpFormatterColon)
class MeasurementStorage:

    plugin_name = None
    klasses = {}

    def __init__(self, path):
        self.path = path

    @classmethod
    def add_storage_klass(kls, klass):
        kls.klasses[klass.plugin_name] = klass

    @classmethod
    def new(kls, dim, path):
        klass = kls.klasses[dim]
        obj = klass(path)
        return obj
        
    @contextlib.contextmanager
    def open(self):
        if self.path and self.path != '-':
            ff = open(self.path, 'wb')
        else:
            ff = sys.stdout
            
        try:
            yield ff
        finally:
            if ff is not sys.stdout:
                ff.close()

    def write(self, cols, data, **meta):
        return NotImplemented


#------------------------------------------------------------------------------
# plugin classes implementing `Algorithm`        
#------------------------------------------------------------------------------
def ReadConfigFile(parser, config_file):
    """
    Read parameters from a file using YAML syntax
    
    The function uses the specified ArgumentParser to:
        * infer default values
        * check if parameter values are consistent with `choices`
        * infer the `type` of each parameter
        * check if any required parameters are missing
    """
    import yaml
    
    # make a new namespace
    ns, unknown = Namespace(), Namespace()

    # read the yaml config file
    config = yaml.load(open(config_file, 'r'))
    
    # first search for plugins
    plugins = []
    if 'X' in config:
        plugins = config['X']
        if isinstance(plugins, str):
            plugins = [plugins]
        for plugin in plugins: load(plugin)
        unknown.X = plugins 
        config.pop('X')

    
    # set defaults, check required and choices
    argnames = set([a.dest for a in parser._actions])
    missing = []
    types = {}
    for a in parser._actions:
        
        # set the default, if not suppressed
        if a.default != SUPPRESS:
            setattr(ns, a.dest, a.default)
            
        # check for choices
        if a.dest in config: 
            if a.choices is not None:
                if config[a.dest] not in a.choices:
                    args = (a.dest, config[a.dest], ", ".join(["'%s'" %x for x in a.choices]))
                    raise ValueError("argument %s: invalid choice '%s' (choose from %s)" %args)
            if a.dest not in types and a.type is not None:
                types[a.dest] = a.type
        else:        
            # track missing        
            if a.required: missing.append(a.dest)
                
    # raise error if missing
    if len(missing):
        missing = "(%s)" % ", ".join("'%s'" %k for k in missing)
        args = (parser.format_usage(), parser.prog, missing)
        raise ValueError("%s\n\n%s: too few arguments, missing: %s" %args)
                
    # set the values, casting if available
    for k in config:
        v = config[k]
        if k in argnames:
            if k in types: v = types[k](v)
            setattr(ns, k, v)
        else:
            setattr(unknown, k, v)
    
    return ns, unknown
    
    
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
        klass = kls.plugins[name]
        
        # get the namespace from the config file
        return ReadConfigFile(klass.parser, config_file)
        

__all__ = ['DataSource', 'Painter', 'Transfer', 'MeasurementStorage', 'Algorithm']

def plugin_isinstance(string, extensionpt):
    """
    Return `True` if the string representation of an extension point
    is an instance of the extension point class `extensionpt`
    """
    if not hasattr(extensionpt, 'plugins'):
        raise TypeError("please specify a valid extension point as the second argument")
        
    if not isinstance(string, str):
        return False
    return string.split(":")[0] in extensionpt.plugins.keys()
