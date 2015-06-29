from . import painters_dir, os
from argparse import ArgumentParser
import imp

#-------------------------------------------------------------------------------
class InputFieldType(object):
    """ Describing an (empty) input field, to be used by the painters 
        plugins in the `painters` directory. Each specific painter plugin
        will `register` its desired field type by adding subparsers
        to this class.
        
        This class is responsible for handling the parsing of arguments 
        that must be passed to each separate painter plugin.
    """
    parser = ArgumentParser("", prefix_chars="-&")
    subparsers = parser.add_subparsers()
    h = subparsers.add_parser("self")
    h.set_defaults(painter=None)

    def __init__(self, string):
        self.string = string
        words = string.split(':')
        ns = self.parser.parse_args(words)
        self.__dict__.update(ns.__dict__)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.string == other.string)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    @classmethod
    def add_parser(kls, name, usage):
        return kls.subparsers.add_parser(name, 
                usage=usage, add_help=False, prefix_chars="&")
    @classmethod
    def format_help(kls):
        rt = []
        for k in kls.subparsers.choices:
            if k != "self":
                rt.append(kls.subparsers.choices[k].format_help())

        if not len(rt):
            return "error: no available input field types"
        else:
            return '\n'.join(rt)
    #-------------------------------------------------------------------------------
def available_field_types():
    """ Return a list of the names of the available field types,
        which have matching painter plugins in the `painters` directory
        
        Naming convention is assumed that for a field type `FieldType`, the
        matching painter plugin must be named `FieldTypePainter`
    
        Return
        ------
        field_types : list
            list of strings giving the names of the available field types
    """
    painters = os.listdir(painters_dir)
    toret = []
    for painter in painters:
        if "Painter" in painter:
            toret.append(painter.split("Painter")[0])
    return toret

#-------------------------------------------------------------------------------
def load_painter_module(field_type):
    """ Import a painter plugin module, returning the module. 
    
        Notes
        -----
        The painter plugin is assumed to have the name `field_type`+`Painter`, 
        and a directory with this name must be located in the `painters`
        directory. The plugin directory must hold a file `__init__.py`
        that will be imported
        
        Parameters
        ----------
        field_type : str
            The name of the field type to load the painter for
    """
    
    # verify that the field type string is valid
    available = available_field_types()
    if field_type not in available:
        args = (field_type, ", ".join("`%s`" %x for x in available))
        raise ValueError("Field type `{}` is not valid; available options: {}".format(*args))
    
    # verify the location of the painter plugin
    location = os.path.join(painters_dir, field_type+"Painter")
    if not os.path.isdir(location):
        args = (field_type+"Painter", painters_dir)
        raise ValueError("{} must be a directory in the directory {}".format(*args))
    if not os.path.exists(os.path.join(location, "__init__.py")):
        raise ValueError("File __init__.py must exist in directory " +
                         "{} for painter plugin import to succeed".format(location))
        
    info = imp.find_module("__init__", [location])
    return imp.load_module("__init__", *info)
    
#-------------------------------------------------------------------------------
def load_painter(field_type):
    """
        Load the painter class, by first importing the painter plugin
        for the specified plugin and then returning the class attribute
    """
    # load the painter module
    mod = load_painter_module(field_type)
    
    # verify the name of the class defined in the above module
    class_name = field_type + "Painter"
    if not hasattr(mod, class_name):
        args = (field_type, class_name)
        raise AttributeError("Painter plugin for `{}` must have class `{}` defined".format(*args))
    
    return getattr(mod, class_name)
    
#-------------------------------------------------------------------------------
def register_field_types(input_field_type):
    """
        Call the `register` class method of each painter plugin class
    """
    # call the `register` method for each painter class
    for field_type in available_field_types():
        painter = load_painter(field_type)
        painter.register(input_field_type)

#-------------------------------------------------------------------------------