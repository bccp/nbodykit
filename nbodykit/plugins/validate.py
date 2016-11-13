import traceback
import inspect
import functools

from .schema import ConstructorSchema

def validate_choices(schema, args_dict):
    """
    Verify that the input values are consistent
    with the `choices`, using the schema
    """
    for attr, val in args_dict.items():
        if attr in schema and schema[attr].choices is not None:
            arg = schema[attr]
            if val not in arg.choices:
                raise ValueError("valid choices for '%s' are: '%s'" %(arg.name, str(arg.choices)))
      
def validate_required_attributes(plugin):
    """
    Validate that the plugin has the required attributes, where
    `plugin` has already been initialized
    
    This looks for the :attr`required_attributes` class attribute
    of plugin.
    
    Parameters
    ---------
    plugin : 
        the initialized plugin instance
    """
    required = getattr(plugin.__class__, 'required_attributes', [])
    
    missing = []
    for name in required:
        if not hasattr(plugin, name):
            missing.append(name)
            
    if len(missing):
        cls = plugin.__class__
        name = getattr(cls, 'plugin_name', cls.__name__)
        args = (name, str(missing))
        raise AttributeError("%s plugin cannot be initialized with missing attributes: %s" %args)

def validate__init__(init):
    """
    Validate the input arguments to :func:`__init__` using
    the class schema
    
    Parameters
    ----------
    init : callable
        the __init__ function we are decorating
    """    
    # inspect the function signature
    attrs, varargs, varkw, defaults = inspect.getargspec(init)
    if defaults is None: defaults = []
             
    # verify the schema
    validate__init__signature(init, attrs, defaults)
         
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
            
        # validate "choices" for positional arguments
        args_dict = dict(zip(attrs[1:], args))
        validate_choices(self.schema, args_dict)    
        
        # validate "choices" for keyword arguments 
        validate_choices(self.schema, kwargs)   
                            
        # call the __init__ to confirm proper initialization
        try:
            init(self, *args, **kwargs)
        except Exception as e:
            
            # get the error message
            errmsg = get_init_errmsg(init.schema, args_dict, kwargs)
            
            # format the total message
            args = (self.__class__.__name__,)
            msg = '\n' + '-'*75 + '\n'
            msg += "error initializing __init__ for '%s':\n" %self.__class__.__name__
            msg += "\t%-25s: '%s'\n %s \n" %("original error message", str(e), traceback.format_exc())
            if len(errmsg): msg += "%s\n" %errmsg
            msg += '-'*75 + '\n'
            e.args = (msg, )
            raise
                
        # validate required attributes
        validate_required_attributes(self)
    
    return wrapper
    
def validate__init__signature(func, attrs, defaults):
    """
    Update the schema, which is attached to `func`,
    using information gathered from the function's signature, 
    namely `attrs` and `defaults`
    
    This will update the `required` and `default` values
    of the arguments of the schema, using the signature of `func`
    
    It also verifies certain aspects of the schema, mostly as a
    consistency check for the developers
    """
    args = attrs[1:] # ignore self

    # get the required names and default names
    required = args; default_names = []
    if defaults is not None and len(defaults):
        required = args[:-len(defaults)]  
        default_names = args[-len(defaults):]
    
    # loop over the schema arguments
    extra = []; missing = default_names + required
    for name in func.schema:
        a = func.schema[name]

        # infer required and defaults and update them
        d = a._asdict()
        d['required'] = a.name in required
        if a.name in default_names:
            d['default'] = defaults[default_names.index(a.name)]

        func.schema[name] = func.schema.Argument(**d)

        # check for extra and missing
        if a.name not in args and not getattr(a, '_hidden', False):
            extra.append(a.name)
        elif a.name in missing:
            missing.remove(a.name)

    # crash if we are missing or got extra (sanity check)
    if len(missing):
        raise ValueError("missing arguments in schema : %s " %str(missing))
    if len(extra):
        raise ValueError("extra arguments in schema : %s" %str(extra))

    # reorder the schema list to match the function signature
    schema_keys = [k for k in func.schema.keys() if k in args]
    if schema_keys != args:
        new_schema = ConstructorSchema(description=func.schema.description)
        for a in args:
            if a in func.schema:
                new_schema[a] = func.schema[a]
        func.schema = new_schema
        
    # update the doc with the schema documentation
    if func.__doc__:
        func.__doc__ += "\n\n" + func.schema.format_help()
    else:
        func.__doc__ = func.schema.format_help()

     
def get_init_errmsg(schema, posargs, kwargs):
    """
    Return a reasonable error message, accounting for:
    
        * missing arguments
        * extra arguments
        * duplicated positional + keyword arguments
    """
    errmsg = ""
    
    # check duplicated
    duplicated = list(set(posargs.keys()) & set(kwargs.keys()))
    if len(duplicated):
        s = "duplicated arguments"
        errmsg += "\t%-25s: %s\n" %(s, str(duplicated))
        
    # check for missing arguments
    required = [s for s in schema if schema[s].required]
    missing = []
    for r in required:
        if r not in posargs and r not in kwargs:
            missing.append(r)
    if len(missing):
        s = "missing arguments"
        errmsg += "\t%-25s: %s\n" %(s, str(missing))
    
    # check for extra arguments
    keys = list(set(posargs.keys()) | set(kwargs.keys()))
    extra = []
    for k in keys:
        if k not in schema:
            extra.append(k)
    if len(extra):
        s = "extra arguments"
        errmsg += "\t%-25s: %s\n" %(s, str(extra))
    
    return errmsg
