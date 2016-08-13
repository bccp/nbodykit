"""
Declare the :class:`PluginMount` base class and various extension points

To define a class as an extension point: 

1. Add a class decorator :meth:`@ExtensionPoint <ExtensionPoint>`

To define a plugin: 

1.  Subclass from the desired extension point class
2.  Define a class method `register` that declares
    the relevant attributes by calling 
    :func:`~nbodykit.utils.config.ConstructorSchema.add_argument` of 
    the class's :class:`~nbodykit.utils.config.ConstructorSchema`, 
    which is stored as the `schema` attribute
3.  Define a `plugin_name` class attribute
4.  Define the functions relevant for that extension point interface.
"""
# MPI will be required because
# a plugin instance will be created for a MPI communicator.
from mpi4py import MPI

# private variable to store global MPI communicator 
# that all plugins are initialized with
_comm = MPI.COMM_WORLD
_cosmo = None

def get_nbkit_comm():
    """
    Return the global MPI communicator that all plugins 
    will be instantiated with (stored in `comm` attribute of plugin)
    """
    return _comm
    
def set_nbkit_comm(comm):
    """
    Set the global MPI communicator that all plugins 
    will be instantiated with (stored in `comm` attribute of plugin)
    """
    global _comm
    _comm = comm
    
def get_nbkit_cosmo():
    """
    Return the global Cosmology instance
    """
    return _cosmo
    
def set_nbkit_cosmo(cosmo):
    """
    Set the global Cosmology instance
    """
    global _cosmo
    _cosmo = cosmo