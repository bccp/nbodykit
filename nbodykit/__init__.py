from .version import __version__

import os
from nbodykit.plugins.manager import PluginManager
from argparse import Namespace

# store some directory paths
pkg_dir      = os.path.abspath(os.path.join(__file__, '..', '..'))
examples_dir = os.path.join(pkg_dir, 'examples')
bin_dir      = os.path.join(pkg_dir, 'bin')

# create the singleton plugin manager, with core plugins loaded
core_paths = [os.path.join(pkg_dir, 'nbodykit', 'core')]
core_paths.append(os.path.join(pkg_dir, 'nbodykit', 'io'))
plugin_manager = PluginManager.create(core_paths, qualprefix='nbodykit')

# create namespaces for the core plugins
algorithms  = Namespace(**plugin_manager['Algorithm'])
datasources = Namespace(**plugin_manager['DataSource'])
transfers   = Namespace(**plugin_manager['Transfer'])
painters    = Namespace(**plugin_manager['Painter'])


class GlobalComm(object):
    """
    The global MPI communicator
    """
    _instance = None
    
    @classmethod
    def get(cls):
        """
        Get the communicator, return ``MPI.COMM_WORLD``
        if the comm has not be explicitly set yet
        """
        # initialize MPI and set the comm if we need to
        if not cls._instance:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            cls._instance = comm
            
        return cls._instance
        
    @classmethod
    def set(cls, comm):
        """
        Set the communicator to the input value
        """
        cls._instance = comm
        
class GlobalCosmology(object):
    """
    The global :class:`~nbodykit.cosmology.Cosmology` instance
    """
    _instance = None 
    
    @classmethod
    def get(cls):
        """
        Get the communicator, return ``MPI.COMM_WORLD``
        if the comm has not be explicitly set yet
        """
        return cls._instance
        
    @classmethod
    def set(cls, cosmo):
        """
        Set the communicator to the input value
        """
        cls._instance = cosmo


            
        
