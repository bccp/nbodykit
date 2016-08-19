from ..extern.six import add_metaclass
from .fromfile import PluginParsingError, EmptyConfigurationError
from . import hooks

from abc import ABCMeta, abstractmethod

def MetaclassWithHooks(meta, *hooks):
    """
    Function to return a subclass of the metaclass `meta`,
    that optionally applies a series of `hooks` when 
    initializing the metaclass
    
    The hooks operate on the parent class of the metaclass,
    allowing the hook functions a method of dynamically modifying
    the parent class 
    
    Parameters
    ----------
    meta : type
        the metaclass that we will subclass
    hooks : list of callables
        functions taking a single argument (the class), which
        can be used to modify the class definition dynamically
    
    Returns
    -------
    wrapped : metaclass
        a subclass of `meta` that applies the specified hooks
    """
    if not len(hooks): return meta
    hooks = getattr(meta, 'hooks', []) + list(hooks)
    
    class wrapped(meta):
        def __init__(cls, name, bases, attrs):
            for hook in hooks: hook(cls)
 
    wrapped.hooks = hooks
    return wrapped      
 
# default hooks: add logger, add schema, use autoassign, attach comm
default_hooks = [hooks.add_logger, hooks.add_and_validate_schema, hooks.attach_comm]
PluginBaseMeta = MetaclassWithHooks(ABCMeta, *default_hooks)

@add_metaclass(PluginBaseMeta)
class PluginBase(object):
    """
    A base class for plugins, designed to be subclassed to implement
    user plugins
    
    The functionality here allows the plugins to be able
    to be loaded from YAML configuration files
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
        
    @property
    def string(self):
        """
        A unique identifier for the plugin, using the :func:`id`
        function
        """
        return id(self)
    
    @classmethod
    @abstractmethod
    def fill_schema(cls):
        """
        The class method responsible fill the class's schema with
        the relevant parameters from the :func:`__init__` signature.
        
        The schema allows the plugin to be initialized properly from
        a configuration file, validating the proper __init__ signatue. 
        
        This should call :func:`~nbodykit.utils.config.ConstructorSchema.add_argument` 
        of the class's :class:`~nbodykit.utils.config.ConstructorSchema`, 
        which is stored as the :attr:`schema` class attribute
        """
        pass
    
    @classmethod
    def create(cls, plugin_name, use_schema=False, **config):
        """
        Instantiate a plugin from this extension type,
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
        from nbodykit import plugin_manager
        name = getattr(cls, 'plugin_name', cls.__name__)
        
        # `plugin_name` must either refer `cls` or a subclass of `cls`
        if plugin_name != name and cls.__name__ not in plugin_manager.supported_types:
            args = (plugin_name, cls.__name__)
            raise ValueError("'%s' does not match the names of any loaded plugins for '%s' class" %args)
           
        # plugin_name refers to cls 
        if name == plugin_name:
            klass = cls
        # plugin_name refers to subclass of cls
        else:
            registry = plugin_manager[cls.__name__]
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
            
                
def ListPluginsAction(extension_type, comm):
    """
    Return a :class:`argparse.Action` that prints
    the help message for the class specified
    by `extension_type`
    
    This action can take any number of arguments. If
    no arguments are provided, it prints the help 
    for all registered plugins of type `extension_type`
    """
    from argparse import Action, SUPPRESS
    
    class ListPluginsAction(Action):
        def __init__(self,
                     option_strings,
                     dest=SUPPRESS,
                     default=SUPPRESS,
                     help=None, 
                     nargs=None,
                     metavar=None):
            Action.__init__(self, 
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=nargs,
                help=help,
                metavar=metavar)
        
        def __call__(self, parser, namespace, values, option_string=None):
            if comm.rank != 0:
                parser.exit(0)
                
            from nbodykit import plugin_manager
            parser.exit(0, plugin_manager.format_help(extension_type, *values))
            
    return ListPluginsAction