from ..core import core_extension_points
import os
import inspect
import traceback
import sys
from collections import defaultdict, OrderedDict
from ..extern.six import string_types

class PluginManager(object):
    """
    Loads and unloads core plugins.
    """
    _instance = None # for using singleton pattern
    supported_types = core_extension_points()

    @classmethod
    def get(cls, *search_dirs):
        """
        Return a PluginManager instance. 
        
        Uses the singleton pattern to ensure that only one
        plugin manager exists
        """
        if not cls._instance:
            PluginManager._instance = cls(*search_dirs)
        return cls._instance

    def __init__(self, *search_dirs):
        """
        This should not be called by the user
        """
        pluginclasses, _ = self.gather_plugins(search_dirs)
        self.__plugins = defaultdict(OrderedDict)
        for plugin_type, p in pluginclasses:
            self.__plugins[plugin_type][p.plugin_name] = p

    def __contains__(self, name):
        
        plugin = None
        try: 
            for extension_type in self.supported_types:
                plugin = self.get_plugin(name, type=extension_type)
        except: 
            pass
        return plugin is not None
            
    def __iter__(self):
        
        for extension_type in self.supported_types:
            yield extension_type, self.__plugins[extension_type]
    
    def __getitem__(self, key):
        
        valid = list(self.supported_types)
        if isinstance(key, string_types):
            if key in valid:
                return self.__plugins[key]
        elif isinstance(key, list):
            if all(k in valid for k in key):
                toret = OrderedDict()
                for k in key: 
                    toret.update(**self.__plugins[k])
            return toret
            
        raise KeyError("key should be a string or list of strings from %s" %str(valid))
    
    
    def find_plugins(self, path):
        """
        Return a list with all plugins found in path
        """
        ext = os.extsep+'py'
        if os.path.isfile(path):
            if not path.endswith(ext):
                raise ValueError("file path for plugins should end in '%s'" %ext)
            files = [path]
        else:
            files = []
            for (dirpath, dirnames, filenames) in os.walk(path):
                files.extend([os.path.join(dirpath, x) for x in filenames if x.endswith(ext)])
                
        plugins = []; modules = []
        for f in files:
            try:
                mod = self._import_file(f)
            except Exception:
                tb = traceback.format_exc()
                print("Importing plugin from %s failed!\n%s" % (f, tb))
                continue
            # get all classes in the imported file
            members = inspect.getmembers(mod, lambda x: inspect.isclass(x))
            # only get classes which are defined, not imported, in mod
            classes = [m[1] for m in members if m[1].__module__ == mod.__name__]
            for c in classes:
                # if the class is derived from a supported type append it
                # we test if it is a subclass of a supported type but not a supported type itself
                # because that might be the abstract class
                for supported_name in self.supported_types:
                    
                    supported = self.supported_types[supported_name]
                    if issubclass(c, supported) and c not in self.supported_types.values():
                        plugins.append((supported.__name__, c))
                        modules.append(mod)
                        
                        # qualify the module name that is loaded by the extension type
                        # this should avoid name collisions
                        if mod.__name__ in sys.modules:
                            sys.modules.pop(mod.__name__)
                        sys.modules['%s.%s' %(supported_name, mod.__name__)] = mod
        return plugins, modules

    def gather_plugins(self, paths):
        """
        Return all plugins that are found in the plugin paths
        """
        plugins = []; modules = []
        # first find built-ins then the ones in the config, then the one from the environment
        # so user plugins can override built-ins
        for p in reversed(paths):
            if p and os.path.exists(p):  # in case of an empty string, we do not search!
                more_plugins, more_modules = self.find_plugins(p)
                plugins += more_plugins; modules += more_modules
        return plugins, modules

    def _import_file(self, f):
        """
        Import the specified file and return the imported module
        """
        directory, module_name = os.path.split(f)
        module_name = os.path.splitext(module_name)[0]

        sys.path.insert(0, directory)
        module = __import__(module_name)
        return module

    def add_user_plugin(self, *paths):
        """
        Dynamically load a user plugin and add it to the specified
        module
        """
        module = "nbodykit.core.user"
        
        # try to import user module
        try:        
            user_mod = __import__(module, fromlist=module.split('.'))
        except Exception:
            raise ImportError("error trying to import destination user module '%s'" %module)
        
        # try to gather user plugins
        plugins, modules = self.gather_plugins(paths)
        for  (plugin_type, p), mod in zip(plugins, modules):
            self.__plugins[plugin_type][p.plugin_name] = p
            setattr(user_mod, mod.__name__, mod)
    
    def get_plugin(self, plugin, type=None):
        """
        Return the plugin instance for the given plugin name
        """
        matches = []
        for extension_type in self.__plugins:
            plugins = self.__plugins[extension_type]
            if plugin in plugins:
                if type is not None and type != extension_type:
                    continue
                matches.append(plugins[plugin])
                
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1 and type is None:
            raise ValueError("name collision for plugin '%s'; please specify 'type'" %plugin)
        else:
            raise ValueError("plugin with name '%s' not found" %plugin)