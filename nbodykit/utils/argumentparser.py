import argparse
from .. import plugins
import re
import sys
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
#        preparser.convert_arg_line_to_args = self.convert_arg_line_to_args
        preparser._read_args_from_files = ArgumentParser._read_args_from_files.__get__(preparser)         
        preparser._yield_args_from_files = ArgumentParser._yield_args_from_files.__get__(preparser)         
        preparser.convert_args_file_to_args = ArgumentParser.convert_args_file_to_args.__get__(preparser)         

        self.ns, unknown = preparser.parse_known_args(args) 

        argparse.ArgumentParser.__init__(self, *largs, **kwargs)

        self.add_argument("-X", action='append', help='path of additional plugins to be loaded' )
 
    def parse_args(self, args=None):
        return argparse.ArgumentParser.parse_args(self, args)

    # override file reading option to treat each line as 
    # an argument and ignore comments. Can put option + value on same line

    def _read_args_from_files(self, arg_strings):
        return list(self._yield_args_from_files(arg_strings))

    def _yield_args_from_files(self, arg_strings):
        # expand arguments referencing files
        for arg_string in arg_strings:
            # for regular arguments, just add them back into the list
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                yield arg_string
                continue
            # replace arguments referencing files with the file content
            try:
                with open(arg_string[1:]) as args_file:
                    for arg in self._yield_args_from_files(
                            self.convert_args_file_to_args(args_file)
                        ):
                        yield arg
            except IOError:
                err = sys.exc_info()[1]
                self.error(str(err))

    def convert_args_file_to_args(self, args_file):
        r""" accepts

             bulh  \    # comments
        """
        store = []
        for line in args_file.readlines():
            if line[0] == '#':
                continue
            r = line.find(' #')
            if r >= 0:
                line = line[:r] 
            r = line.find('\t#')
            if r >= 0:
                line = line[:r] 

            line = line.strip()
            if len(line) == 0: continue

            if line[-1] == '\\':
                line = line[:-1].strip()
                if len(line) > 0:
                    store.append(line)
            else:
                line = line.strip()
                if len(line) > 0:
                    store.append(line)
                if len(store) > 0:
                    yield ''.join(store)
                store = []
        if len(store) > 0: 
            yield ''.join(store)

