from abc import abstractmethod
from ...plugins import PluginBase

class Transfer(PluginBase):
    """
    Mount point for plugins which apply a k-space transfer function
    to the Fourier transfrom of a datasource field
    
    Plugins of this type should provide the following attributes:

    plugin_name : str
        class attribute that defines the name of the plugin in 
        the registry
    
    register : classmethod
        a class method taking no arguments that updates the
        :class:`~nbodykit.utils.config.ConstructorSchema` with
        the arguments needed to initialize the class
    
    __call__ : method
        function that will apply the transfer function to the complex array
    """
    @abstractmethod
    def __call__(self, pm, complex):
        """ 
        Apply the transfer function to the complex field
        
        Parameters
        ----------
        pm : ParticleMesh
            the particle mesh object which holds possibly useful
            information, i.e, `w` or `k` arrays
        complex : array_like
            the complex array to apply the transfer to
        """
        pass