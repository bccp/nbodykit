import inspect
import functools
from collections import namedtuple 

def create_plugin(cls, plugin_name, cosmo, **kwargs):
    """
    Instantiate and return a Plugin, directly from the name of 
    the Plugin and the necessary attributes, passed as keywords
    
    If the Plugin is a DataSource, also pass the `cosmo` instance
    to the Plugin initialization
    """
    if cls.__name__ == 'DataSource':
        kwargs.update(cosmo=cosmo)
            
    return cls.create(plugin_name, **kwargs)
            
def initialize_plugins(d, cosmo=None):
    """
    Recursively search a parsed YAML output, replacing
    plugin names and arguments with the initialized instances
    
    """
    from nbodykit.extensionpoints import isplugin, get_extensionpt
    
    # check for strings that represent Plugins (with no attributes)
    if isinstance(d, str) and isplugin(d):
        cls = get_extensionpt(d)
        return create_plugin(cls, d, cosmo)
        
    # if not a dict/list, just return it
    if not isinstance(d, (dict, list)):
        return d
        
    # if a dictionary with `plugin` key, make a plugin
    if isinstance(d, dict) and 'plugin' in d:
        kwargs = d.copy()
        name = kwargs.pop('plugin')
        cls = get_extensionpt(name)
        return create_plugin(cls, name, cosmo, **kwargs)
            
    # loop over the list/dict and recursively search
    for i, k in enumerate(d):
               
        # check for plugins
        if isinstance(k, str) and isplugin(k):
            cls = get_extensionpt(k)
            
            # make plugin from (key, value) = (plugin, arguments)
            if isinstance(d, dict):
                            
                kwargs = d[k].copy() if d[k] is not None else {}
                plugin = create_plugin(cls, k, cosmo, **kwargs)
                
                # new key for this plugin is name of ExtensionPoint
                d.pop(k); k = cls.__name__
                if k in d:
                    if isinstance(d[k], list):
                        d[k].append(plugin)
                    else:
                        d[k] = [d[k], plugin]
                else:
                    d[k] = plugin
                
            # make plugin from just the key (no arguments)
            else:
                d[i] = create_plugin(cls, k, cosmo)
         
        # call recursively   
        if isinstance(d, dict):
            d[k] = initialize_plugins(d[k], cosmo)
        else:
            d[i] = initialize_plugins(d[i], cosmo)
        
    return d

import yaml
from collections import OrderedDict

def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    """
    Load from yaml into OrderedDict to preserve the ordering used 
    by the user
    
    see: http://stackoverflow.com/questions/5121931/
    """
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ReadConfigFile(config_file, schema):
    """
    Read parameters from a file using YAML syntax
    
    The function uses the specified `schema` to:
        * infer default values
        * check if parameter values are consistent with `choices`
        * infer the `type` of each parameter
        * check if any required parameters are missing
    """
    from argparse import Namespace
    from nbodykit.cosmology import Cosmology
    
    # make a new namespace
    ns, unknown = Namespace(), Namespace()

    # read the yaml config file
    config = ordered_load(open(config_file, 'r'))
    
    # first search for plugins
    plugins = []
    if 'X' in config:
        plugins = config['X']
        if isinstance(plugins, str):
            plugins = [plugins]
        for plugin in plugins: load(plugin)
        unknown.X = plugins 
        config.pop('X')
    
    # now load cosmology
    cosmo = None
    if 'cosmo' in config:
        cosmo = Cosmology(**config['cosmo'])
        config.pop('cosmo')

    # initialize plugins
    config = initialize_plugins(config, cosmo)
                
    # set the values, casting if available
    for k in config:
        v = config[k]
        if k in schema:
            cast = schema[k].type
            if cast is not None: v = cast(v)
            setattr(ns, k, v)
        else:
            setattr(unknown, k, v)
    
    return ns, unknown

class ConstructorSchema(list):
    """
    A list of named tuples, which store the relevant argument
    information needed to construct a class.
    
    Notes
    -----
    *   you can test whether an argument 'name' is in the schema
        with the usual syntax:
            >> param_name = 'param'
            >> contains = param_name in schema
            
    *   in addition to the normal `list` syntax, you can access
        the named tuple arguments via a dict-like syntax
            >> arg_tuple = schema[param_name]
    """
    Argument = namedtuple('Argument', ['name', 'type', 'default', 'choices', 'help', 'required'])
    
    def __init__(self, *args, **kwargs):
        super(ConstructorSchema, self).__init__(*args)
        self.description = kwargs.get('description', "")
     
    def __contains__(self, key):
        return key in [a.name for a in self]
    
    def __getitem__(self, key):
        
        if isinstance(key, str):
            names = [a.name for a in self]
            if key not in names:
                raise KeyError("no such argument with name '%s'" %key)
            return self[names.index(key)]
        else:
            return list.__getitem__(self, key)
    
    def add_argument(self, name, type=None, default=None, choices=None, help=None, required=True):
        """
        Add an argument to the schema
        """        
        arg = self.Argument(name, type, default, choices, help, required)
        self.append(arg)
        
    def format_help(self):
        """
        Return a string giving the help 
        """
        toret = ""
        if getattr(self, 'description', ""):
            toret += self.description + '\n\n'
            
        optional = []; required = []
        for p in self:
            h = p.help if p.help is not None else ""
            info = "  %-20s %s" %(p.name, h)
            if p.default is not None:
                info += " (default: %s)" %p.default
            if p.required:
                required.append(info)
            else:
                optional.append(info)
            
        toret += "required arguments:\n%s\n" %('-'*18)
        toret += "\n".join(required)
        toret += "\n\noptional arguments:\n%s\n" %('-'*18)
        toret += "\n".join(optional)
        return toret
        
def attribute(name, **kwargs):
    """
    Declare a class attribute, adding it to the schema attached to 
    the function we are decorating
    """
    def _argument(func):
        if not hasattr(func, 'schema'):
            func.schema = ConstructorSchema()
        func.schema.add_argument(name, **kwargs)
        return func
    return _argument
    
def autoassign(init, allowed=[]):
    """
    Verify the schema attached to the input `init` function,
    automatically set the input arguments, and then finally
    call `init`
    """
    # inspect the function
    attrs, varargs, varkw, defaults = inspect.getargspec(init)
    if defaults is None: defaults = []
    
    # verify the schema
    update_schema(init, attrs, defaults, ignore=allowed)
    
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        
        # handle extra allowed keywords (that aren't in signature)
        for k in allowed:
            val = kwargs.pop(k, init.schema[k].default)
            setattr(self, k, val)
        
        # handle default values
        for attr, val in zip(reversed(attrs), reversed(defaults)):
            setattr(self, attr, val)
        
        # handle positional arguments
        positional_attrs = attrs[1:]            
        for attr, val in zip(positional_attrs, args):
            check_choices(init.schema, attr, val)
            setattr(self, attr, val)
    
        # handle varargs
        if varargs:
            remaining_args = args[len(positional_attrs):]
            setattr(self, varargs, remaining_args)                
        
        # handle varkw
        if kwargs:
            for attr,val in kwargs.items():
                check_choices(init.schema, attr, val)
                setattr(self, attr, val)
        return init(self, *args, **kwargs)
    return wrapper
    
def check_choices(schema, attr, val):
    """
    Verify that the input values are consistent
    with the `choices`, using the schema
    """
    if attr in schema:
        arg = schema[attr]
        if arg.choices is not None:
            if val not in arg.choices:
                raise ValueError("valid choices for '%s' are: '%s'" %(arg.name, str(arg.choices)))

def cast(schema, attr, val):
    """
    Convenience function to cast values based
    on the `type` stored in `schema`, and check `choices`
    """
    if attr in schema:
        arg = schema[attr]
        if arg.type is not None: 
            val = arg.type(val)
        if arg.choices is not None:
            if val not in arg.choices:
                raise ValueError("valid choices for '%s' are: '%s'" %(arg.name, str(arg.choices)))
    return val

def update_schema(func, attrs, defaults, ignore=[]):
    """
    Update the schema, which is attached to `func`,
    using information gather from the function's signature, 
    namely `attrs` and `defaults`
    
    This will update the `required` and `default` values
    of the schema, using the signature of `func`
    
    It also verifies certain aspects of the schema, mostly as a
    consistency check on the developer
    """
    args = attrs[1:] # ignore self
    
    # get the required names and default names
    required = args; default_names = []
    if defaults is not None and len(defaults):
        required = args[:-len(defaults)]  
        default_names = args[-len(defaults):]
    
    # loop over the schema arguments
    extra = []; missing = default_names + required
    for i, a in enumerate(func.schema):
            
        # infer required and defaults and update them
        d = a._asdict()
        d['required'] = a.name in required
        if a.name in default_names:
            d['default'] = defaults[default_names.index(a.name)]
        func.schema[i] = func.schema.Argument(**d)
        
        # check for extra and missing
        if a.name not in args and a.name not in ignore:
            extra.append(a.name)
        elif a.name in missing:
            missing.remove(a.name)

    # crash if we are missing or got extra (sanity check)
    if len(missing):
        raise ValueError("missing arguments in schema : %s " %str(missing))
    if len(extra):
        raise ValueError("extra arguments in schema : %s" %str(extra))

    # reorder the schema list to match the function signature
    order = [args.index(a.name) for a in func.schema if a.name in args]
    N = len(func.schema)-len(ignore)
    if not all(i == order[i] for i in range(N)):
        new_schema = [func.schema[order.index(i)] for i in range(N)]
        for p in ignore: new_schema.append(func.schema[p])
        func.schema = ConstructorSchema(new_schema, description=func.schema.description)
        
        
    # update the doc with the schema documentation
    if func.__doc__:
        func.__doc__ += "\n\n" + func.schema.format_help()
    else:
        func.__doc__ = func.schema.format_help()

