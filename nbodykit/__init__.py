from nbodykit.pluginmanager import load_builtins
import os

load_builtins()

# save the path of a few packages
pkg_dir      = os.path.abspath(os.path.join(__file__, '..', '..'))
examples_dir = os.path.join(pkg_dir, 'examples')
bin_dir      = os.path.join(pkg_dir, 'bin')


try:
    from .version import version as __version__
except ImportError:
    raise ImportError('nbodykit not properly installed. If you are running from '
                      'the source directory, please install it in-place by running: '
                      'pip install -e .')
