from __future__ import print_function
import os
import sys
from nbodykit import extensionpoints

dirname = 'plugins-list'
if not os.path.exists(dirname):
    os.makedirs(dirname)

valid = ['Algorithm', 'DataSource', 'Painter', 'Transfer']
for expoint_str in valid:

    expoint = extensionpoints.extensionpoints[expoint_str]
    name = expoint.__name__
    registry = vars(expoint.registry)

    filename = os.path.join(dirname, name+'.rst')
    print("making %s..." %filename)
    with open(filename, 'w') as ff:

        for k in sorted(registry):
            name = str(registry[k])[8:-2]
            desc = registry[k].schema.description.replace("\n", "")
            line = "- :class:`%s <%s>`: %s\n" %(k, name, desc)
            ff.write(line)
