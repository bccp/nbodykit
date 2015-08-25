from .. import PluginMount

class FileFormat:
    """
    Mount point for plugins which refer to the reading and writing of files.

    Plugins implementing this reference should provide the following 
    attributes:

    field_type : str
        class attribute giving the name of the subparser which 
        defines the necessary command line arguments for the plugin
    
    register : classmethod
        A class method taking no arguments that adds a subparser
        and the necessary command line arguments for the plugin
    
    read: method
        A method that performs the reading of the field. It 
        takes the following arguments:
            ns : argparse.Namespace
            column :
            start :
            end : 
    
    write: method
        A method that performs the writing of the field. It 
        takes the following arguments:
            ns : argparse.Namespace
            column :
            start :
            data :
    """
    __metaclass__ = PluginMount
    formats = {}
 
    def __init__(self, words):
        pass

    @classmethod
    def parse(kls, string): 
        words = string.split(':')
        self.string = string
        klass = words[0].strip()
        klass = kls.formats[klass]
        return klass(words[1:])

    def __eq__(self, other):
        return self.string == other.string

    def __ne__(self, other):
        return self.string != other.string

    def read(self, ns, start, end):
        return NotImplemented    

    def write(self, ns, start, data):
        return NotImplemented    

    @classmethod
    def usage(kls):
        return ""

    @classmethod
    def register(kls):
        FileFormat.formats[kls.__name__] = kls
    
    @classmethod
    def format_help(kls):
        rt = []
        for name in kls.formats:
            plugin = kls.formats[name]
            rt.append(plugin.usage())

        if not len(rt):
            return "No available file types"
        else:
            return '\n'.join(rt)
