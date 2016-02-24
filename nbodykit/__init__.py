from nbodykit.plugins import load
import os.path
path = os.path.abspath(os.path.dirname(__file__))

builtins = ['datasource', 'painter/', 'transfer/', 'algorithms/']

for plugin in builtins:
    load(os.path.join(path, 'plugins', plugin))

 
