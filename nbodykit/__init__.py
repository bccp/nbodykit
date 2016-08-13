__version__ = "0.1.4.dev0"

import os
pkg_dir      = os.path.abspath(os.path.join(__file__, '..', '..'))
examples_dir = os.path.join(pkg_dir, 'examples')
bin_dir      = os.path.join(pkg_dir, 'bin')

# get the singleton plugin manager, with core plugins loaded
from nbodykit.plugins.manager import PluginManager

core_paths = [os.path.join(pkg_dir, 'nbodykit', 'core')]
plugin_manager = PluginManager.get(core_paths, qualprefix='nbodykit')

# create namespaces for the core plugins
from argparse import Namespace

algorithms  = Namespace(**plugin_manager['Algorithm'])
datasources = Namespace(**plugin_manager['DataSource'])
transfers   = Namespace(**plugin_manager['Transfer'])
painters    = Namespace(**plugin_manager['Painter'])
