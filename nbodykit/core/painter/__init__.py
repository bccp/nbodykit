from abc import abstractmethod
from ...plugins import PluginBase

class Painter(PluginBase):
    """
    Mount point for plugins which refer to the painting of data, i.e.,
    gridding a field to a mesh

    Plugins of this type should provide the following attributes:

    plugin_name : str
        A class attribute that defines the name of the plugin in 
        the registry

    register : classmethod
        A class method taking no arguments that updates the
        :class:`~nbodykit.utils.config.ConstructorSchema` with
        the arguments needed to initialize the class

    paint : method
        A method that performs the painting of the field.
    """    
    @abstractmethod
    def __init__(self, paintbrush):
        self.paintbrush = paintbrush
    
    @abstractmethod
    def paint(self, pm, datasource):
        """ 
        Paint the DataSource specified to a mesh

        Parameters
        ----------
        pm : :class:`~pmesh.particlemesh.ParticleMesh`
            particle mesh object that does the painting
        datasource : DataSource
            the data source object representing the field to paint onto the mesh

        Returns
        -------
        stats : dict
            dictionary of statistics related to painting and reading of the DataSource
        """
        pass

    def basepaint(self, real, position, paintbrush, weight=None):
        """
        The base function for painting that is used by default. 
        This handles the domain decomposition steps that are necessary to 
        complete before painting.

        Parameters
        ----------
        pm : :class:`~pmesh.particlemesh.ParticleMesh`
            particle mesh object that does the painting
        position : array_like
            the position data
        paintbrush : string
            picking the paintbrush. Available ones are from documentation of pm.RealField.paint().
        weight : array_like, optional
            the weight value to use when painting

        """
        assert real.pm.comm == self.comm # pm must be from the same communicator!
        layout = real.pm.decompose(position)

        position = layout.exchange(position)
        if weight is not None:
            weight = layout.exchange(weight)
            real.paint(position, weight, method=paintbrush, hold=True)
        else:
            real.paint(position, method=paintbrush, hold=True)