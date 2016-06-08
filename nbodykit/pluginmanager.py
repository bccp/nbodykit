import os
import os.path
import glob
import traceback
from argparse import Action, SUPPRESS
from sys import modules
from types import ModuleType

# import the parent modules
import nbodykit.plugins
import nbodykit.plugins.algorithms
import nbodykit.plugins.datasource
import nbodykit.plugins.painter
import nbodykit.plugins.transfer
import nbodykit.plugins.user

def load(filename):
    return load2(filename, qualifiedprefix='nbodykit.plugins.user')

def load_builtins():
    path = os.path.abspath(os.path.dirname(__file__))
    return load2(os.path.join(path, 'plugins'), qualifiedprefix='nbodykit')

def load2(filename, qualifiedprefix='nbodykit.plugins.user'):
    """ load a plugin from filename.

        Parameters
        ----------
        filename : string
            path to the .py file
        qualifiedprefix: string
            a prefix to build a qualified name in sys.modules. This
            is used to load the built-in plugins in nbodykit.plugins

        Returns
        -------
        namespace : dict
            modified global namespace of the plugin script.
    """
    if os.path.isdir(filename):
        root = filename.rstrip('/')
        qualname = '.'.join([qualifiedprefix, os.path.basename(root)])
        if qualname not in modules:
            __import__(qualname)
        module = modules[qualname]
        for filename in sorted(os.listdir(root)):
            fullfilename = os.path.join(root, filename)
            basename = os.path.splitext(os.path.basename(filename))[0]
            add = False
            if os.path.isdir(fullfilename):
                add = True
            if filename.endswith('.py'):
                add = True
            if basename.startswith('__'):
                continue

            if add:
                module.__dict__[basename] = load2(fullfilename, qualifiedprefix=qualname)
        return module
    basename, ext = os.path.splitext(os.path.basename(filename))
    qualname = '.'.join([qualifiedprefix, basename])
    module = ModuleType(qualname)

    with open(filename, 'r') as f:
        code = compile(f.read(), filename, 'exec')
        exec(code, module.__dict__)

    modules[qualname] = module
    return module

def ListPluginsAction(extensionpoint):
    class ListPluginsAction(Action):
        def __init__(self,
                     option_strings,
                     dest=SUPPRESS,
                     default=SUPPRESS,
                     help=None, 
                     nargs=None,
                     metavar=None):
            Action.__init__(self, 
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=nargs,
                help=help,
                metavar=metavar)
        
        def __call__(self, parser, namespace, values, option_string=None):
            parser.exit(0, extensionpoint.format_help(*values))
            
    return ListPluginsAction
