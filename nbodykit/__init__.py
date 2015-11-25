from nbodykit.plugins import load
import os.path
path = os.path.abspath(os.path.dirname(__file__))

builtins = ['datasource/', 'plugins/Measurement1DStorage.py', 'plugins/Measurement2DStorage.py']

for plugin in builtins:
    load(os.path.join(path, plugin))
 
