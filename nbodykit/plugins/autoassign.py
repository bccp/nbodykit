import inspect
import functools

from .schema import ConstructorSchema
       
def attribute(name, **kwargs):
    """
    A function decorator that adds an argument to the 
    class's schema
    
    See :func:`ConstructorSchema.add_argument` for further details
    
    Parameters
    ----------
    name : the name of the argument to add
    **kwargs : dict
        the additional keyword values to pass to ``add_argument``
    """
    def _argument(func):
        if not hasattr(func, 'schema'):
            func.schema = ConstructorSchema()
        func.schema.add_argument(name, **kwargs)
        return func
    return _argument
    
def autoassign(init):
    """
    Verify the schema attached to the input `init` function,
    automatically set the input arguments, and then finally
    call `init`
    
    Parameters
    ----------
    init : callable
        the function we are decorating
    """
    # inspect the function signature
    attrs, varargs, varkw, defaults = inspect.getargspec(init)
    if defaults is None: defaults = []
             
    # verify the schema
    update_schema(init, attrs, defaults)
         
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
                        
        # handle default values
        for attr, val in zip(reversed(attrs), reversed(defaults)):
            setattr(self, attr, val)
        
        # handle positional arguments
        positional_attrs = attrs[1:]
        posargs = {}            
        for attr, val in zip(positional_attrs, args):
            check_choices(init.schema, attr, val)
            posargs[attr] = val
            setattr(self, attr, val)
    
        # handle varargs
        if varargs:
            remaining_args = args[len(positional_attrs):]
            setattr(self, varargs, remaining_args)            
        
        # handle varkw
        if kwargs:
            for attr,val in kwargs.items():
                check_choices(init.schema, attr, val)
                try: setattr(self, attr, val)
                except: pass
        
        # call the __init__ to confirm proper initialization
        try:
            return init(self, *args, **kwargs)
        except Exception as e:
            
            # get the error message
            errmsg = get_init_errmsg(init.schema, posargs, kwargs)
            
            # format the total message
            args = (self.__class__.__name__,)
            msg = '\n' + '-'*75 + '\n'
            msg += "error initializing __init__ for '%s':\n" %self.__class__.__name__
            msg += "\t%-25s: '%s'\n" %("original error message", str(e))
            if len(errmsg): msg += "%s\n" %errmsg
            msg += '-'*75 + '\n'
            e.args = (msg, )
            raise
            
    return wrapper
    
def update_schema(func, attrs, defaults):
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