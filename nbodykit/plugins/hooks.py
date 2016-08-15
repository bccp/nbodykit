from ..extern.six import PY2
import functools

def _get_init(cls):
    """
    Utility function to return the __init__ function
    of the input class
    """
    # do nothing if __init__ is abstract
    if getattr(cls.__init__, '__isabstractmethod__', False):
        return None
        
    # get the __init__ function
    init = cls.__init__
    if PY2: init = init.__func__
    
    return init


def attach_cosmo(cls):
    """
    A hook that attaches the `cosmo` keyword to a class.
    This class performs two operations:
    
        1.  it adds 'cosmo' to the class schema
        2.  it sets the 'cosmo' keyword argument, using the 
            global value as the default
    """
    # get the __init__
    init = _get_init(cls)
    if init is None: return

    # add to the schema
    if 'cosmo' not in cls.schema:
        h = 'the `Cosmology` class relevant for the DataSource'
        cls.schema.add_argument("cosmo", default=None, help=h)
        cls.schema['cosmo']._hidden = True
    
    # wrap the old __init__ to handle 'cosmo' keyword
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        
        from nbodykit import GlobalCosmology
        setattr(self, 'cosmo', kwargs.pop('cosmo', GlobalCosmology.get()))        
        return init(self, *args, **kwargs)

    cls.__init__ = wrapper


def attach_comm(cls):
    """
    A hook that attaches the `comm` keyword to a class.
    This class performs two operations:
    
        1.  it adds 'comm' to the class schema
        2.  it sets the 'comm' keyword argument, using the 
            global value as the default
    """
    # get the __init__
    init = _get_init(cls)
    if init is None: return

    # add to the schema
    if 'comm' not in cls.schema:
        h = 'the global MPI communicator'
        cls.schema.add_argument("comm", default=None, help=h)
        cls.schema['comm']._hidden = True
    
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
                        
        from nbodykit import GlobalComm
        setattr(self, 'comm', kwargs.pop('comm',  GlobalComm.get()))
        return init(self, *args, **kwargs)

    cls.__init__ = wrapper


def add_logger(cls):
    """
    A hook that adds a logger to the input class as
    a class attribute :attr:`logger`
    """
    import logging
    name = getattr(cls, 'plugin_name', cls.__name__)
    cls.logger = logging.getLogger(name)


def add_and_validate_schema(cls):
    """
    A hook that: 
    
        1.  adds a ConstructorSchema to the input class
        2.  calls the :func:`fill_schema` class method, if available
        3.  decorates __init__ to validate arguments at the time of 
            intialization
    """
    from .schema import ConstructorSchema
    from .validate import validate__init__
    
    # get the __init__
    init = _get_init(cls)
    if init is None: return
    
    # add a schema
    init.schema = ConstructorSchema()
    cls.schema = cls.__init__.schema
    
    # call the fill_schema function
    if hasattr(cls, 'fill_schema'):
        cls.fill_schema()
        
    # validate the __init__ arguments
    cls.__init__ = validate__init__(init)    