from abc import abstractmethod
from ...plugins import PluginBase

class Algorithm(PluginBase):
    """
    Mount point for plugins which provide an interface for running
    one of the high-level algorithms, i.e, power spectrum calculation
    or FOF halo finder
    
    Plugins of this type should provide the following attributes:

    plugin_name : str
        A class attribute that defines the name of the plugin in 
        the registry
    
    register : classmethod
        A class method taking no arguments that updates the
        :class:`~nbodykit.utils.config.ConstructorSchema` with
        the arguments needed to initialize the class
    
    run : method
        function that will run the algorithm
    
    save : method
        save the result of the algorithm computed by :func:`Algorithm.run`
    """  
    @abstractmethod      
    def run(self):
        """
        Run the algorithm
        
        Returns
        -------
        result : tuple
            the tuple of results that will be passed to :func:`Algorithm.save`
        """
        pass
    
    @abstractmethod
    def save(self, output, result):
        """
        Save the results of the algorithm run
        
        Parameters
        ----------
        output : str
            the name of the output file to save results too
        result : tuple
            the tuple of results returned by :func:`Algorithm.run`
        """
        pass
