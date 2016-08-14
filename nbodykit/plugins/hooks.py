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


def add_schema(cls):
    """
    A hook to add a ConstructorSchema to the input class
    and call the :func:`register` class method, if 
    available
    """
    from .schema import ConstructorSchema
    
    # get the __init__
    init = _get_init(cls)
    if init is None: return
    
    # add a schema
    init.schema = ConstructorSchema()
    cls.schema = cls.__init__.schema
    
    # call the register function
    if hasattr(cls, 'register'):
        cls.register()
        
        
def autoassign(cls):
    """
    A hook to call the `autoassign` decorator on the 
    input class's :func:`__init__` function
    
    The `autoassign` decorator reads the signature of
    the :func:`__init__` function and automatically
    sets the input attributes before calling the
    :func:`__init__` function
    """
    from .autoassign import autoassign as decorator
    
    # get the __init__
    init = _get_init(cls)
    if init is None: return

    # configure the class __init__
    cls.__init__ = decorator(init)
    