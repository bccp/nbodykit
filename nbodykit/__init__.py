__version__ = "0.1.3"

from nbodykit.pluginmanager import load_builtins
import os

load_builtins()

# save the path of a few packages
pkg_dir      = os.path.abspath(os.path.join(__file__, '..', '..'))
examples_dir = os.path.join(pkg_dir, 'examples')
bin_dir      = os.path.join(pkg_dir, 'bin')
