import os.path
import glob
import traceback
from argparse import Action, SUPPRESS

references = {}

def load(filename, namespace=None):
    """ load a plugin from filename.
        
        Parameters
        ----------
        filename : string
            path to the .py file
        namespace : dict
            global namespace, if None, an empty space will be created
        
        Returns
        -------
        namespace : dict
            modified global namespace of the plugin script.
    """
    if os.path.isdir(filename):
        l = glob.glob(os.path.join(filename, "*.py"))
        for f in l:
            load(f, namespace)
        return
    if namespace is None:
        namespace = {}
    namespace = dict(namespace)
    try:
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')
            exec(code, namespace)
    except Exception as e:
        raise RuntimeError("Failed to load plugin '%s': %s : %s" % (filename, str(e), traceback.format_exc()))
    references[filename] = namespace
    return filename

def ListPluginsAction(extensionpoint):
    class ListPluginsAction(Action):
        def __init__(self,
                     option_strings,
                     dest=SUPPRESS,
                     default=SUPPRESS,
                     help=None, 
                     nargs=None):
            Action.__init__(self, 
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=nargs,
                help=help)
        
        def __call__(self, parser, namespace, values, option_string=None):
            parser.exit(0, extensionpoint.format_help(*values))
            
    return ListPluginsAction