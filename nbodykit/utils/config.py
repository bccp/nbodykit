import inspect
import functools
from collections import namedtuple 

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
        Return a string of help 
        """     
        toret = "Parameters:\n%s\n" %('-'*10)
        s = []
        for p in self:
            h = p.help if p.help is not None else ""
            info = "  %-20s %s" %(p.name, h)
            if p.default is not None:
                info += " (default: %s)" %p.default
            s.append(info)
            
        toret += "\n".join(s)
        return toret
        
def Argument(name, **kwargs):
    """
    Add a named argument to the schema attached to the function
    we are decorating
    """
    def _argument(func):
        if not hasattr(func, 'schema'):
            func.schema = ConstructorSchema()
        func.schema.add_argument(name, **kwargs)
        return func
    return _argument
    
def Configure(init):
    """
    Verify the schema attached to the input `init` function,
    automatically set the input arguments, and then finally
    call `init`
    """
    # inspect the function
    attrs, varargs, varkw, defaults = inspect.getargspec(init)
    
    # verify the schema
    update_schema(init, attrs, defaults)
    
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        # handle default values
        for attr, val in zip(reversed(attrs), reversed(defaults)):
            setattr(self, attr, val)
        
        # handle positional arguments
        positional_attrs = attrs[1:]            
        for attr, val in zip(positional_attrs, args):
            val = cast(init.schema, attr, val)
            setattr(self, attr, val)
    
        # handle varargs
        if varargs:
            remaining_args = args[len(positional_attrs):]
            setattr(self, varargs, remaining_args)                
        
        # handle varkw
        if kwargs:
            for attr,val in kwargs.items():
                val = cast(init.schema, attr, val)
                setattr(self, attr, val)
        return init(self, *args, **kwargs)
    return wrapper
    
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

def update_schema(func, attrs, defaults):
    """
    Update the schema, which is attached to `func`,
    using information gather from the function's signature, 
    namely `attrs` and `defaults`
    
    This will update the `required` and `default` values
    of the schema, using the signature of `func`
    
    It also verifies certain aspects of the schema, mostly as a
    consistency check on the developer
    """
    args = attrs[1:]
    required = args; default_names = []
    if len(defaults):
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
        if a.name not in args:
            extra.append(a.name)
        elif a.name in missing:
            missing.remove(a.name)

    # crash if we are missing or got extra (sanity check)
    if len(missing):
        raise ValueError("missing arguments in schema : %s " %str(missing))
    if len(extra):
        raise ValueError("extra arguments in schema : %s" %str(extra))

    # reorder the schema list to match the function signature
    order = [args.index(a.name) for a in func.schema]
    N = len(func.schema)
    if not all(i == order[i] for i in range(N)):
        func.schema = ConstructorSchema([func.schema[order.index(i)] for i in range(N)])
        
    # update the doc with the schema documentation
    if func.__doc__:
        func.__doc__ += "\n\n" + func.schema.format_help()
    else:
        func.__doc__ = func.schema.format_help()

