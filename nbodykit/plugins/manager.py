from ..core import core_extension_points
from ..io import io_extension_points
from ..extern.six import string_types

import os
import sys
import inspect
from types import ModuleType
from collections import defaultdict, OrderedDict

class PluginManager(object):
    """
    A class to manage the loading of plugins in :mod:`nbodykit`.
    
    .. note::
    
        This class uses the ingleton pattern, so only one
        instance exists when :mod:`nbodykit` is loaded
    
        It should be accessed using the :func:`get` function.
    """
    _instance = None # for using singleton pattern
    supported_types = core_extension_points()
    supported_types.update(**io_extension_points())

    @classmethod
    def create(cls, paths, qualprefix="nbodykit.core.user"):
        """
        Create the PluginManager instance
        
        Uses the singleton pattern to ensure that only one
        plugin manager exists
        
        Raises
        ------
        ValueError : 
            if the PluginManager already exists
        
        Parameters
        ----------
        paths : tuple of str
            the search paths to look for the core plugins
        qualprefix : str, optional
            the prefix to build a qualified name in ``sys.modules``. 
            This is used to load the builtin plugins in :mod:`nbodykit.core`
        """
        if cls._instance:
            raise ValueError("PluginManager instance already exists; use PluginManager.get() to access")
        
        PluginManager._instance = cls(paths, qualprefix=qualprefix)
        return cls._instance
        
    @classmethod
    def get(cls):
        """
        Return the PluginManager
        
        Raises
        ------
        ValueError : 
            if the PluginManager has not been created yet
        """
        if not cls._instance:
            raise ValueError("PluginManager instance has not been created yet; see PluginManager.create()")
        return cls._instance

    def __init__(self, paths, qualprefix="nbodykit.core.user"):
        """
        Initialize a new PluginManager from the specified 
        search paths
        
        The user should use :func:`get` to get the instance
        of the PluginManager
        """
        # inititalize the dict to store the plugins
        # this is a dict of OrderedDicts for each supported type
        self.__plugins = defaultdict(OrderedDict)
        
        # load the plugins
        for p in paths:
            self._load(p, qualprefix=qualprefix)

    def __len__(self):
        """
        Return the number of loaded plugins
        """
        return sum(len(self.__plugins[k]) for k in self.supported_types)
    
    def __repr__(self):
        args = (self.__class__.__name__, len(self))
        return "<%s: %d loaded plugins>" %args
        
    def __contains__(self, name):
        """
        Check if a plugin name has been loaded by the 
        PluginManager
        """
        if not isinstance(name, string_types):
            return False
        
        plugin = None
        try: 
            for extension_type in self.supported_types:
                plugin = self.get_plugin(name, type=extension_type)
        except: 
            pass
        return plugin is not None
            
    def __iter__(self):
        """
        Yield tuples of (extension type, dict of plugins)
        """
        for extension_type in self.supported_types:
            yield extension_type, self.__plugins[extension_type]
    
    def __getitem__(self, key):
        """
        Return the dict of plugins for the input extension type
        """
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
    
    def _load(self, fn, qualprefix):
        """ 
        Load plugins from the specified filename

        Parameters
        ----------
        fn : string
            path to the .py file
        qualprefix: string
            a prefix to build a qualified name in sys.modules. This
            is used to load the built-in plugins in nbodykit.plugins
        """
        # load the directory tree
        if os.path.isdir(fn):
            
            # get the qualified module (importing if we need to)
            root = fn.rstrip(os.sep)
            qualname = '.'.join([qualprefix, os.path.basename(root)])
            if qualname not in sys.modules:
                __import__(qualname)
            module = sys.modules[qualname]
            
            # loop over files in the root directory
            for fn in sorted(os.listdir(root)):
                
                fullfn = os.path.join(root, fn)
                basename = os.path.splitext(os.path.basename(fn))[0]
                
                # skip __init__
                if basename.startswith('__'):
                    continue
                    
                # recursively load plugins in the directory
                # and add them to the module
                if os.path.isdir(fullfn) or fn.endswith(os.path.extsep + 'py'):
                    mod = self._load(fullfn, qualname)
                    self._extract_plugins(mod)
                    module.__dict__[basename] = mod

            return module
        
        # load the normal file into a module
        basename, ext = os.path.splitext(os.path.basename(fn))
        qualname = '.'.join([qualprefix, basename])
        
        # only add the new module if its not there
        # strange behavior in python 2 if you overwrite modules
        # see http://goo.gl/vMIU3v
        if qualname not in sys.modules:
            
            # make the new module object
            module = ModuleType(qualname)
            
            # execute the code
            with open(fn, 'r') as f:
                code = compile(f.read(), fn, 'exec')
                exec(code, module.__dict__)
           
            sys.modules[qualname] = module
            self._extract_plugins(module)
        else:
            module = sys.modules[qualname]
            
        return module
        
    def _extract_plugins(self, mod):
        """
        Given a module, extract the relevant plugin classes and
        save them to the PluginManager
        
        Parameters
        ----------
        mod : module
            the module to search for classes
        """
        # get classes defined in this module
        members = inspect.getmembers(mod, lambda x: inspect.isclass(x))
        classes = [m[1] for m in members if m[1].__module__ == mod.__name__]

        # save classes that are subclasses of a supported type
        s = self.supported_types
        for c in classes:
            for name in s:
                supported = s[name]
                if issubclass(c, supported) and c not in s.values():
                    self.__plugins[name][c.plugin_name] = c        

    def add_user_plugin(self, *paths):
        """
        Dynamically load user plugins, adding each plugin
        to :mod:`nbodykit.core.user`
        
        .. note::
            
            At the moment, loaded plugins must be a subclass
            of one of the classes defined in :attr:`supported_types`
        
        Parameters
        ----------
        paths : tuple of str
            the file paths to search for plugins to load
        """
        for p in paths:
            self._load(p, qualprefix='nbodykit.core.user')
    
    def get_plugin(self, name, type=None):
        """
        Return the plugin instance for the given plugin name
        
        Parameters
        ----------
        name : str
            the name of the plugin to load
        type : str, optional
            the extension type (one of :attr:`supported_types`);
            in the case of name collisions across different 
            extension types, this parameter must be given, otherwise,
            an exception will be raised
        
        Returns
        -------
        cls
            the plugin class corresponding to `name`
        """
        matches = []
        for extension_type in self.__plugins:
            plugins = self.__plugins[extension_type]
            if name in plugins:
                if type is not None and type != extension_type:
                    continue
                matches.append(plugins[name])
                
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1 and type is None:
            raise ValueError("name collision for plugin '%s'; please specify 'type'" %name)
        else:
            raise ValueError("plugin with name '%s' not found" %name)
            
    def format_help(self, extension, *plugins):
        """
        Return a string specifying the `help` for each of the plugins
        specified of extension type `extension`
        
        If no `plugins` are specified, format the help message for 
        all plugins registered as subclasses of `extension`
        
        Parameters
        ----------
        extension : str
            the string specifying the extension type; should be in 
            :attr:`PluginManager.supported_types`
        plugins : list of str
            strings specifying the names of plugins of the specified type
            to format together
        
        Returns
        -------
        help_msg : str
            the formatted help message string for the specified plugins
        """
        if extension not in self.supported_types:
            raise ValueError("`extension` argument should be one of %s" %list(self.supported_types))
        
        registry = self[extension]
        if not len(plugins):
            plugins = list(registry)
            
        s = []
        for k in plugins:
            
            if k not in registry:
                raise ValueError("'%s' is not a valid plugin name for type '%s'" %(k, extension))
            
            header = "Plugin : %s  Extension Type : %s" % (k, extension)
            s.append(header)
            s.append("=" * (len(header)))
            s.append(registry[k].schema.format_help() + "\n")

        if not len(s):
            return "No available plugins registered for type '%s'" %extension
        else:
            return '\n'.join(s) + '\n'