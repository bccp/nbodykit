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
class PluginInterface(object):
    """ The basic interface of a plugin 
    """
    def initialize(self, args):
        ns = self.parser.parse_args(args)
        self.__dict__.update(ns.__dict__)

    @classmethod
    def register(kls):
        raise NotImplementedError

    def __eq__(self, other):
        return self.string == other.string

    def __ne__(self, other):
        return self.string != other.string


def ExtensionPoint(cls):
    """ Declares a class as an extension point """
    return add_metaclass(PluginMount)(cls)

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

            if cls.plugin_name in cls.plugins:
                raise RuntimeError("Plugin class %s already registered with %s"
                    % (cls.plugin_name, str(type(cls))))

            # add a commandline argument parser that parsers the ':' seperated
            # commandlines.
            cls.parser = ArgumentParser(cls.plugin_name, 
                    usage=None, add_help=False, 
                    formatter_class=HelpFormatterColon)

            # track names of classes
            cls.plugins[cls.plugin_name] = cls
            
            # try to call register class method
            if hasattr(cls, 'register'):
                cls.register()

    def create(kls, string): 
        """ Instantiate a plugin from this extension point,
            based on the cmdline string

            Parameters
            ----------
            string: string
                A colon (:) separated string of arguments.
                The first field specifies the type of the plugin
                to create.
                The reset depends on the type of the plugin.
        """
        words = string.split(':')
        
        klass = kls.plugins[words[0]]
        
        self = klass()
        self.initialize(words[1:])
        self.string = string
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

import numpy
from nbodykit.plugins import HelpFormatterColon
from argparse import ArgumentParser

@ExtensionPoint
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

@ExtensionPoint
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
        raise NotImplementedError

    def read(self, columns, comm, stat, full=False):
        """ 
            Yield the data in the columns. If full is True, read all
            particles in one run; otherwise try to read in chunks.

            On every iteration stat is updated with the global 
            statistics. Current keys are min, max, Ntot.
            
        """
        if comm.rank == 0:
            data = self.readall(columns)    
            shape_and_dtype = [(d.shape, d.dtype) for d in data]
            Ntot = len(data[0])
        else:
            shape_and_dtype = None
            Ntot = None
        shape_and_dtype = comm.bcast(shape_and_dtype)
        stat['Ntot'] = comm.bcast(Ntot)

        if comm.rank != 0:
            data = [
                numpy.empty(0, dtype=(dtype, shape[1:]))
                for shape,dtype in shape_and_dtype
            ]

        yield data 

import numpy
from nbodykit.plugins import HelpFormatterColon
from argparse import ArgumentParser

@ExtensionPoint
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

        for data in datasource.read(columns, pm.comm, stats, full=False):
            data = dict(zip(columns, data))
            position = data['Position']

            layout = pm.decompose(position)

            for c in list(data.keys()):
                data[c] = layout.exchange(data[c])
                
            yield [data[c] for c in columns]

import sys
import contextlib

@ExtensionPoint
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
        
__all__ = ['DataSource', 'Painter', 'Transfer', 'MeasurementStorage']

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
    
