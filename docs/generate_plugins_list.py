import os
import sys
import mock
MOCK_MODULES = ['mpsort', 'mpi4py', 'scipy', 'scipy.interpolate', 'h5py', 'bigfile', 
                'kdcount', 'pmesh', 'pmesh.particlemesh', 'pmesh.domain', 'pytest',
                'astropy']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

from nbodykit import extensionpoints
dirname = 'plugins-list'
if not os.path.exists(dirname):
    os.makedirs(dirname)
    
valid = extensionpoints.__valid__
for expoint in valid:
    
    name = expoint.__name__
    registry = vars(expoint.registry)
    
    filename = os.path.join(dirname, name+'.rst') 
    print "making %s..." %filename
    with open(filename, 'w') as ff:
        
        for k in sorted(registry):
            name = str(registry[k])[8:-2]
            desc = registry[k].schema.description.replace("\n", "")
            line = "- :class:`%s <%s>`: %s\n" %(k, name, desc)
            ff.write(line)
        