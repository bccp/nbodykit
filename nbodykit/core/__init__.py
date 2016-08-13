from nbodykit.core.algorithms import Algorithm
from nbodykit.core.datasource import DataSource, GridSource, DataStream
from nbodykit.core.painter import Painter
from nbodykit.core.transfer import Transfer

__all__ = ['Algorithm', 'DataSource', 'GridSource', 'Painter', 'Transfer']

def core_extension_points():
    g = globals()
    return {k:g[k] for k in __all__}