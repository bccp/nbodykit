import traceback
from collections import namedtuple, OrderedDict
from ..extern.six import string_types

# base class for arguments of a schema is a named tuple
fields = ['name', 'required', 'type', 'default', 'choices', 'nargs', 'help', 'subfields']
ArgumentBase = namedtuple('Argument', fields)

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

class Argument(ArgumentBase):
    """
    Class to represent an argument in the `ConstructorSchema`
    """
    def __new__(cls, name, required, type=None, default=None, choices=None, nargs=None, help="", subfields=None):
        if subfields is None: subfields = OrderedDict()
        return super(Argument, cls).__new__(cls, name, required, type, default, choices, nargs, help, subfields)

    def __getitem__(self, key):
        if isinstance(key, string_types):
            return self.subfields[key]
        return super(Argument, self).__getitem__(key)

class ConstructorSchema(OrderedDict):
    """
    A subclass of :class:`~collections.OrderedDict` to hold 
    :class:`Argument` objects, with argument names as the keys of 
    the dictionary.
    
    Each :class:`Argument` stores the relevant information of that
    argument, included `type`, `help`, `choices`, etc. 
    
    Each :class:`Argument` also stores a `subfields` attribute, which
    is a new :class:`~collections.OrderedDict` of `Argument` objects
    to store any sub-fields
    
    Notes
    -----
    You can test whether a full argument 'name' is in the schema
    with the `contains` function
    
    >> argname = 'field.DataSource'
    >> schema.contains(argname)
    True
            
    Arguments that are subfields can be accessed in a sequential 
    dict-like fashion:
    
    >> subarg = schema['field']['DataSource']
    """  
    # store the Argument class as a class variable
    Argument = Argument 
              
    def __init__(self, description=""):
        """
        Initialize an empty schema, optionally passing a 
        description of the schema
        
        Parameters
        ----------
        description : str, optional
            the string description of the schema
        """
        super(ConstructorSchema, self).__init__()
        self.description = description
    
    def __repr__(self):
        """
        A string representation that outputs total number of parameters
        and how many of those parameters are optional
        """
        size = len(self)
        required = sum(1 for k in self if self[k].required)            
        args = (self.__class__.__name__, size, size-required)
        return "<%s: %d parameters (%d optional)>" %args

    @staticmethod
    def cast(arg, value):
        """
        Convenience function to cast values based
        on the `type` stored in `schema`. 
        
        .. note::
            
            If the `type` of `arg` is a tuple, then each 
            type will be attempted in the order given.

        Parameters
        ----------
        arg : :class:`Argument`
            the `Argument` which gives the relevant meta-data
            to properly cast `value`
        value : 
            the value we are casting, using the `type`
            attribute of `arg`
        
        Returns
        -------
        casted : 
            the casted value; can have many types
        """
        # if we expect a list, make value a list
        if arg.nargs is not None:
            if not isinstance(value, list): value = [value]
            
        # check that number of arguments is required if nargs is specified
        if isinstance(arg.nargs, int) and len(value) != arg.nargs:
            raise ValueError("'%s' requires exactly %d arguments" %(arg.name, arg.nargs))
        
        # require one or more input values
        if arg.nargs == '+' and len(value) == 0:
            raise ValueError("'%s' requires at least one argument" %arg.name)

        def _cast(cast):
            if cast is None: return value
            if arg.nargs is not None:
                r = [cast(v) for v in value]
            else:
                r = cast(value)
            return r

        # should be a tuple of casting callables
        casts = arg.type
        if not isinstance(casts, tuple):
            casts = (casts,)
        
        # try each cast
        exceptions = []
        for cast in casts:
            try:
                return _cast(cast)
            except Exception as e:
                exceptions.append(str(e) + traceback.format_exc())

        # if we get here, no casts worked
        msg = "\n".join("\t%d) %s" %(i+1,e) for i,e in enumerate(exceptions))
        raise ValueError("unable to successfully cast parameter '%s'; exceptions:\n%s" %(arg.name, msg))
        
    def contains(self, key):
        """
        Check if the schema contains the full argument name, using
        `.` to represent subfields
        
        Examples
        --------
        >> argname = 'field.DataSource'
        >> schema.contains(argname)
        True
        
        Parameters
        ----------
        key : str
            the name of the argument to search for
        """
        split = key.split('.')
        prefix = split[:-1]; name = split[-1]
        obj = self
        for k in prefix:
            obj = self[k].subfields
        
        return name in obj 
    
    def add_argument(self, name, type=None, default=None, choices=None, nargs=None, help=None, required=False):
        """
        Add an argument to the schema
        
        Parameters
        ----------
        name : str
            the name of the parameter to add
        type : callable, optional
            a function that will cast the parsed value
        default : optional
            the default value for this parameter
        choices : optional
            the distinct values that the parameter can take
        nargs : int, '*', '+', optional
             the number of arguments that should be consumed for this parameter
        help : str, optional
            the help string
        required : bool, optional
            whether the parameter is required or not    
        """                
        # get the prefix
        split = name.split('.')
        prefix = split[:-1]; suffix = split[-1]
        
        # create default parent Arguments that do not exist
        obj = self
        for i, k in enumerate(prefix):
            if k not in obj:
                obj[k] = Argument('.'.join(prefix[:i+1]), required)
            obj = obj[k].subfields
        
        # add new argument (with empty subfields)
        if not self.contains(name):
            obj[suffix] = Argument(name, required, nargs=nargs, type=type, default=default, 
                                    choices=choices, help=help)
                                    
        # overwrite existing object (copying the subfields)
        else:
            obj[suffix] = obj[suffix]._replace(type=type, default=default, choices=choices, 
                                                help=help, required=required, nargs=nargs)
     
    def _arg_info(self, name, arg, level, subfield=False):
        """
        Internal helper function that returns the info string 
        for one argument, indenting to match a specific level
        
        Format: name: description (default=`default`)
        """  
        indent = " "*4
        space = indent*level
        
        # determine the string representation of the type 
        if arg.choices is not None:
            type_str = "{ %s }" %", ".join(["'%s'" %str(s) for s in arg.choices])
        else:
            casts = arg.type
            if not isinstance(arg.type, tuple):
                casts = (casts, )

            type_str = []
            for cast in casts:
                cstr = cast.__name__ if arg.type is not None else ""
                if hasattr(cast, '__self__'):
                    cstr = '.'.join([cast.__self__.__name__, cast.__name__])

                # don't use function names when it's a lambda function
                if 'lambda' in cstr: cstr = ""
                type_str.append(cstr)
            type_str = ', '.join(type_str)

        # optional tag?
        if not subfield and not arg.required:
            if type_str: type_str += ", "
            type_str += 'optional'
            
        # first line is name : type, indented `level` times
        info = "%s%s : %s\n" %(space, name, type_str)
        
        # second line gives the description, indented `level+1` times
        info += "%s%s" %(indent*(level+1), arg.help)
        if arg.default is not None:
            info += " (default: %s)" %arg.default
        return info
                     
    def _parse_info(self, name, arg, level, subfield=False):
        """
        Internal function to recursively parse a argument and any
        subfields, returning the full into string
        """
        indent = " "*4
        info = self._arg_info(name, arg, level, subfield=subfield)
        if not len(arg.subfields):
            return info
        
        info += "\n" + indent*(level) + "    " + "The %d subfields are:" %(len(arg.subfields))
        info += '\n'
        for k in arg.subfields:
            v = arg.subfields[k]
            info += '\n'+self._parse_info(k, v, level+2, subfield=True)
        info += '\n'
             
        return info
            
    def format_help(self):
        """
        Return a string giving the help using the 
        format preferred by the ``numpy`` documentation
        """
        toret = """"""
        if getattr(self, 'description', ""):
            toret += self.description + '\n\n'
            
        optional = []; required = []
        for k in self:
            arg = self[k]
            info = self._parse_info(k, arg, level=0)
            if arg.required:
                required.append(info)
            else:
                optional.append(info)
            
        toret += "Parameters\n----------\n"
        toret += "\n".join(required + optional)
        return toret
        
    __str__ = format_help
