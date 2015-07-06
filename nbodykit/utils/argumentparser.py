import argparse
from .. import plugins
import re

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *largs, **kwargs):
        kwargs['formatter_class'] = argparse.RawTextHelpFormatter
        kwargs['fromfile_prefix_chars']="@"
        args = kwargs.pop('args', None)
            
        preparser = argparse.ArgumentParser(add_help=False, 
                fromfile_prefix_chars=kwargs['fromfile_prefix_chars'])
        preparser.add_argument("-X", type=plugins.load, action="append")
        # Process the plugins
        preparser.exit = lambda a, b: None
        preparser.convert_arg_line_to_args = self.convert_arg_line_to_args
         
        self.ns, unknown = preparser.parse_known_args(args) 

        argparse.ArgumentParser.__init__(self, *largs, **kwargs)

        self.add_argument("-X", action='append', help='path of additional plugins to be loaded' )
    
    def parse_args(self, args=None):
        return argparse.ArgumentParser.parse_args(self, args)

    # override file reading option to treat each space-separated word as 
    # an argument and ignore comments. Can put option + value on same line

    def convert_arg_line_to_args(self, line):
        r = line.find(' #')
        if r >= 0:
            line = line[:r] 
        r = line.find('\t#')
        if r >= 0:
            line = line[:r] 
        yield line

