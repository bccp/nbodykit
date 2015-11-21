from nbodykit.plugins import load
import os.path
path = os.path.abspath(os.path.dirname(__file__))

builtins = ['datasource/', 'plugins/Power1DStorage.py', 'plugins/Power2DStorage.py']

for plugin in builtins:
    load(os.path.join(path, plugin))
 
