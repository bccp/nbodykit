from __future__ import print_function
import os
import sys
from nbodykit import plugin_manager

dirname = 'plugins-list'
if not os.path.exists(dirname):
    os.makedirs(dirname)

for expoint_str in plugin_manager.supported_types:

    expoint = plugin_manager.supported_types[expoint_str]
    name = expoint.__name__
    registry = plugin_manager[expoint_str]

    filename = os.path.join(dirname, name+'.rst')
    print("making %s..." %filename)
    with open(filename, 'w') as ff:

        for k in sorted(registry):
            name = str(registry[k])[8:-2]
            desc = registry[k].schema.description.replace("\n", "")
            line = "- :class:`%s <%s>`: %s\n" %(k, name, desc)
            ff.write(line)
