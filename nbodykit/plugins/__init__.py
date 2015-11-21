"""
    Declare PluginMount and various extention points.

    To define a Plugin, set __metaclass__ to PluginMount, and
    define a .register member. For Python 3 and 2 compatibilibty
    use add_metaclass decorator.

"""

class PluginMount(type):
    
    def __init__(cls, name, bases, attrs):

        # only executes when processing the mount point itself.
        if not hasattr(cls, 'plugins'):
            cls.plugins = {}
        # called for each plugin, which already has 'plugins' list
        else:
            if not hasattr(cls, 'field_type'):
                raise RuntimeError("Plugin class must carry a field_type.")

            if cls.field_type in cls.plugins:
                raise RuntimeError("Plugin class %s already registered with %s"
                    % (cls.field_type, str(type(cls))))

            # track names of classes
            cls.plugins[cls.field_type] = cls
            
            # try to call register class method
            if hasattr(cls, 'register'):
                cls.register()

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
@add_metaclass(PluginMount)
class DataSource:
    """
    Mount point for plugins which refer to the reading of input files 
    and the subsequent painting of those fields.

    Plugins implementing this reference should provide the following 
    attributes:

    field_type : str
        class attribute giving the name of the subparser which 
        defines the necessary command line arguments for the plugin
    
    register : classmethod
        A class method taking no arguments that adds a subparser
        and the necessary command line arguments for the plugin
    
    paint : method
        A method that performs the painting of the field. It 
        takes the following arguments:
            pm : pypm.particlemesh.ParticleMesh

    read: method
        A method that performs the reading of the field. This method
        reads in the full data set. It shall
        returns the position (in 0 to BoxSize) and velocity (in the
        same units as position)

    read_comm: method
        A method that performs the reading of the field. It shall
        returns the position (in 0 to BoxSize) and velocity (in the
        same units as position), in chunks as an iterator. The
        default behavior is to use Rank 0 to read in the full data
        and yield an empty data.

    """
    
    field_type = None

    def __init__(self, args):
        ns = self.parser.parse_args(args)
        self.__dict__.update(ns.__dict__)

    @classmethod
    def open(kls, connection): 
        """ opens a file based on the connection string 

            Parameters
            ----------
            connection: string
                A colon (:) separated string of arguments.
                The first field is the type of the connection.
                The reset depends on the type of the conntection.
        """
        words = connection.split(':')
        
        klass = kls.plugins[words[0]]
        self = klass(words[1:])
        self.string = connection
        return self

    def __eq__(self, other):
        return self.string == other.string

    def __ne__(self, other):
        return self.string != other.string

    def readall(self, columns):
        return NotImplemented 

    def read(self, columns, comm, bunchsize):
        """ 
            Yield the data in the columns.
        """
        if comm.rank == 0:
            data = self.readall(columns)    
            shape_and_dtype = [(d.shape, d.dtype) for d in data]
        else:
            shape_and_dtype = None
        shape_and_dtype = comm.bcast(shape_and_dtype)

        if comm.rank != 0:
            data = [
                numpy.empty(0, dtype=(dtype, shape[1:]))
                for shape,dtype in shape_and_dtype
            ]

        yield data 

    @classmethod
    def add_parser(kls):
        from ..utils.pluginargparse import HelpFormatterColon
        from argparse import ArgumentParser
        kls.parser = ArgumentParser(kls.field_type, 
                usage=None, add_help=False, formatter_class=HelpFormatterColon)
        return kls.parser

    @classmethod
    def format_help(kls):
        
        rt = []
        for k in kls.plugins:
            rt.append(kls.plugins[k].parser.format_help())

        if not len(rt):
            return "No available input field types"
        else:
            return '\n'.join(rt)

#------------------------------------------------------------------------------
import sys
import contextlib

@add_metaclass(PluginMount)
class PowerSpectrumStorage:

    field_type = None
    klasses = {}

    def __init__(self, path):
        self.path = path

    @classmethod
    def add_storage_klass(kls, klass):
        kls.klasses[klass.field_type] = klass

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

    def write(self, data, **meta):
        return NotImplemented

#------------------------------------------------------------------------------          
import os.path
import glob

references = {}

def load(filename, namespace=None):
    """ load a plugin from filename.
        
        Parameters
        ----------
        filename : string
            path to the .py file
        namespace : dict
            global namespace, if None, an empty space will be created
        
        Returns
        -------
        namespace : dict
            modified global namespace of the plugin script.
    """
    if os.path.isdir(filename):
        l = glob.glob(os.path.join(filename, "*.py"))
        for f in l:
            load(f, namespace)
        return
    if namespace is None:
        namespace = {}
    namespace = dict(namespace)
    try:
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')
            exec(code, namespace)
    except Exception as e:
        raise RuntimeError("Failed to load plugin '%s': %s" % (filename, str(e)))
    references[filename] = namespace

