from abc import ABCMeta, abstractmethod
from ..extern.six import add_metaclass
from nbodykit.plugins.config import PluginParsingError, make_configurable
import argparse 
    
def ABCMetaWithHooks(*hooks):
    class wrapped(ABCMeta):
        def __init__(cls, name, bases, attrs):
            try:
                for hook in hooks: hook(cls)
            except Exception as e:
                raise
                pass
    return wrapped

@add_metaclass(ABCMetaWithHooks(make_configurable))
class PluginBase(object):
    """
    A plugin that can be loaded from an input configuration file
    """
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def registry(cls):
        toret = {}
        for c in cls.__subclasses__():
            name = getattr(c, 'plugin_name', c.__name__)
            toret[name] = c

        return toret
    
    @classmethod
    @abstractmethod
    def register(cls):
        pass
    
    @classmethod
    def create(cls, plugin_name, use_schema=False, **config):
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
            if `True`, cast the keywords that are defined in 
            the class schema before initializing. Default: `False`
        **config : dict
            the parameter names and values that will be
            passed to the plugin's __init__

        Returns
        -------
        plugin : 
            the initialized instance of `plugin_name`
        """
        registry = cls.registry()
        name = getattr(cls, 'plugin_name', cls.__name__)
        if plugin_name != name and plugin_name not in registry:
            raise ValueError("'%s' does not match the names of any loaded plugins for '%s' class" %(plugin_name, str(cls)))
            
        if name == plugin_name:
            klass = cls
        else:
            klass = registry[plugin_name]
        
        # cast the input values, using the class schema
        if use_schema:
            if not hasattr(klass, 'schema'):
                raise PluginParsingError("nonexistent plugin schema -- cannot use info to cast")
            for k in config:
                if k in klass.schema:
                    arg = klass.schema[k]
                    config[k] = klass.schema.cast(arg, config[k])
                        
        return klass(**config)
        
    @classmethod
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
                        raise PluginParsingError
                elif hasattr(cls, 'plugin_name'):
                    return cls.create(cls.plugin_name, use_schema=True, **parsed)
                else:
                    raise PluginParsingError
            elif isinstance(parsed, str):
                return cls.create(parsed)
            else:
                raise PluginParsingError
        except PluginParsingError as e:
            msg = '\n' + '-'*75 + '\n'
            msg += "failure to parse plugin from configuration using `from_config()`\n"
            msg += ("\nThere are several ways to initialize plugins from configuration files:\n"
                    "1. The plugin parameters are specified as a dictionary containing the key `plugin`,\n"
                    "   which gives the name of the plugin to load; the rest of the dictionary is\n"
                    "   passed to the plugin `__init__()` as keyword arguments\n"
                    "2. The plugin is specified as a dictionary having only one entry -- \n"
                    "   the key gives the plugin name and the value is a dict of arguments\n"
                    "   to be passed to the plugin `__init__()`\n"
                    "3. When `from_config()` is bound to a particular plugin class, only a dict\n"
                    "   of the `__init__()` arguments should be specified\n"
                    "4. The plugin is specified as a string, which gives the name of the plugin;\n"
                    "   the plugin will be created with no arguments\n")
            msg += '\n' + '-'*75 + '\n'
            e.args = (msg,)
            raise 
        except:
            raise
            
    @classmethod 
    def format_help(cls, *plugins):
        """
        Return a string specifying the `help` for each of the plugins
        specified, or all if none are specified
        """
        # dict (name, cls) for each registered plugins of this class type
        registry = cls.registry()
        
        if not len(plugins):
            plugins = list(registry)
            
        s = []
        for k in plugins:
            if not isplugin(k):
                raise ValueError("'%s' is not a valid plugin name" %k)
            header = "Plugin : %s  ExtensionPoint : %s" % (k, cls.__name__)
            s.append(header)
            s.append("=" * (len(header)))
            s.append(registry[k].schema.format_help())

        if not len(s):
            return "No available plugins registered at %s" %cls.__name__
        else:
            return '\n'.join(s) + '\n'
            
def isplugin(name):
    """
    Return `True`, if `name` is a registered plugin for
    any extension point
    """
    return name in PluginBase.registry()
