"""
    Declare PluginMount and various extention points.

    To define a Plugin, set __metaclass__ to PluginMount, and
    define a .register member.

"""

class PluginMount(type):
    
    def __init__(cls, name, bases, attrs):

        # only executes when processing the mount point itself.
        if not hasattr(cls, 'plugins'):
            cls.plugins = []
        # called for each plugin, which already has 'plugins' list
        else:
            # track names of classes
            cls.plugins.append(cls)
            
            # try to call register class method
            if hasattr(cls, 'register'):
                cls.register()

#------------------------------------------------------------------------------
class InputPainter:
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
            ns : argparse.Namespace
            pm : pypm.particlemesh.ParticleMesh
    """
    __metaclass__ = PluginMount
    
    from argparse import ArgumentParser

    parser = ArgumentParser("", add_help=False)
    subparsers = parser.add_subparsers()
    field_type = None

    def __init__(self, dict):
        self.__dict__.update(dict)

    @classmethod
    def parse(kls, string): 
        words = string.split(':')
        
        ns = kls.parser.parse_args(words)
        klass = ns.klass
        d = ns.__dict__
        # break the cycle
        del d['klass']
        d['string'] = string
        painter = klass(d)
        return painter

    def __eq__(self, other):
        return self.string == other.string

    def __ne__(self, other):
        return self.string != other.string

    def paint(self, ns, pm):
        return NotImplemented    

    @classmethod
    def add_parser(kls, name, usage):
        return kls.subparsers.add_parser(name, 
                usage=usage, add_help=False)
    
    @classmethod
    def format_help(kls):
        
        rt = []
        for plugin in kls.plugins:
            k = plugin.field_type
            rt.append(kls.subparsers.choices[k].format_help())

        if not len(rt):
            return "No available input field types"
        else:
            return '\n'.join(rt)

#------------------------------------------------------------------------------
import sys
import contextlib

class PowerSpectrumStorage:
    __metaclass__ = PluginMount

    field_type = None
    klasses = {}

    def __init__(self, path):
        self.path = path

    @classmethod
    def add_storage_klass(kls, klass):
        kls.klasses[klass.field_type] = klass

    @classmethod
    def get(kls, dim, path):
        klass = kls.klasses[dim]
        obj = klass(path)
        return obj
        
    @contextlib.contextmanager
    def open(self):
        if self.path and self.path != '-':
            ff = open(self.path, 'w')
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

def load(filename, namespace=None):
    """ An adapter for ArgumentParser to load a plugin.
        
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
    if namespace is None:
        namespace = {}
    if os.path.isdir(filename):
        # FIXME: walk the dir and load all .py files.
        raise ValueError("Can not load directory")
    try:
        execfile(filename, namespace)
    except Exception as e:
        raise RuntimeError("Failed to load plugin '%s': %s" % (filename, str(e)))
    return namespace


builtins = ['TPMSnapshotPainter', 'HaloFilePainter', 'PandasPlainTextPainter',
            'PlainTextPainter', 'HDFPainter', 'Power1DStorage', 'Power2DStorage', 'fileformats/tpm']
for plugin in builtins:
    globals().update(load(os.path.join(os.path.dirname(__file__), plugin + '.py')))
 
