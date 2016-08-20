from ...plugins import PluginBase, PluginBaseMeta, MetaclassWithHooks
from ...plugins.hooks import attach_cosmo
from ...extern.six import add_metaclass

from abc import abstractmethod, abstractproperty
import numpy

# attach the cosmology to data sources
SourceMeta = MetaclassWithHooks(PluginBaseMeta, attach_cosmo)

@add_metaclass(SourceMeta)
class Source(PluginBase):

    @abstractproperty
    def columns(self):
        return []

    @abstractproperty
    def attrs(self):
        return {}

    @abstractmethod
    def read(self, columns):
        yield []

    @abstractmethod
    def paint(self, pm):
        pass
