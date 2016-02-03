"""
    Declare PluginMount and various extention points.

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

from nbodykit.plugins import HelpFormatterColon
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

algorithms = Namespace()
datasources = Namespace()
painters = Namespace()
transfers = Namespace()
mstorages = Namespace()

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

    def __init__(self, comm, **kwargs):
        self.comm = comm

        # FIXME: set unique string by default
        # directly created object does not have a string!
        self.string = str(id(self)) 

        argnames = set([action.dest for action in self.parser._actions])
        missing = []
        d = {}
        for argname in list(argnames):
            if argname not in kwargs:
                if not action.required:
                    d[argname] = action.default
                else:
                    missing += argname 
            else:
                d[argname] = kwargs[argname]
            argnames.remove(argname)

        if len(missing):
            raise ValueError("Missing arguments : %s " % str(missing))
        if len(argnames):
            raise ValueError("Extra arguments : %s " % str(argnames))

        self.__dict__.update(kwargs)

def ExtensionPoint(registry):
    """ Declares a class as an extension point, registering to registry """
    def wrapped(cls):
        return add_metaclass(PluginMount, registry)(cls)
    return wrapped

class PluginMount(type):
    """ Metaclass for an extension point that provides
        the methods to manage
        plugins attached to the extension point.
    """
    def __new__(cls, name, bases, attrs):
        # for python 2, ensure extension points are objects
        # this is important for python 3 compatibility.
        if len(bases) == 0:
            bases = (object,)
        # Only add PluginInterface to the ExtensionPoint,
        # such that Plugins will inherit from this.
        if len(bases) == 1 and bases[0] is object:
            bases = (PluginInterface,)
        return type.__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, attrs):

        # only executes when processing the mount point itself.
        if not hasattr(cls, 'plugins'):
            cls.plugins = {}
        # called for each plugin, which already has 'plugins' list
        else:
            if not hasattr(cls, 'plugin_name'):
                raise RuntimeError("Plugin class must carry a plugin_name.")
            
            # register, if this plugin isn't yet
            if cls.plugin_name not in cls.plugins:
                # add a commandline argument parser that parsers the ':' seperated
                # commandlines.
                cls.parser = ArgumentParser(cls.plugin_name, 
                        usage=None, add_help=False, 
                        formatter_class=HelpFormatterColon)

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

    def create(kls, argv, comm=None): 
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

            comm: MPI.Comm or None
                The communicator this plugin is instantialized for.
                if None, MPI.COMM_WORLD is assumed.

            Notes
            -----
            2nd stage parsing: A plugin can override 
            `finalize_attribute` to finalize the
            attribute values based on currently parsed attribute values.

            The `comm` attribute stores the communicator for which 
            this plugin is instantialized.

        """
        klass = kls.plugins[argv[0]]
        
        ns = klass.parser.parse_args(argv[1:])
        
        if comm is None:
            comm = MPI.COMM_WORLD

        self = klass(comm, **vars(ns))
        self.string = str(argv)
        self.finalize_attributes()
        return self

    def format_help(kls):
        
        rt = []
        for k in kls.plugins:
            rt.append(kls.plugins[k].parser.format_help())

        if not len(rt):
            return "No available Plugins registered at %s" % kls.__name__
        else:
            return '\n'.join(rt)

# copied from six
def add_metaclass(metaclass, registry):
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
        orig_vars['registry'] = registry
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

    @classmethod
    def fromstring(kls, string, comm=None): 
        argv = string.split(':')
        return kls.create(argv, comm)

@ExtensionPoint(datasources)
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
    def fromstring(kls, string, comm=None): 
        argv = string.split(':')
        return kls.create(argv, comm)

import numpy
from nbodykit.plugins import HelpFormatterColon
from argparse import ArgumentParser

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

            for c in list(data.keys()):
                data[c] = layout.exchange(data[c])
                
            yield [data[c] for c in columns]

    @classmethod
    def fromstring(kls, string, comm=None): 
        argv = string.split(':')
        return kls.create(argv, comm)

import sys
import contextlib

@ExtensionPoint(mstorages)
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
    def run(self, comm=None):
        raise NotImplementedError
    
    def save(self, *args, **kwargs):
        raise NotImplementedError
        

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
    
