import os.path
import glob
import traceback
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

from argparse import ArgumentParser as BaseArgumentParser
from argparse import RawTextHelpFormatter

import re
import sys

    
class ArgumentParser(BaseArgumentParser):
    """ An argument parser that loads plugins before dropping to
        the second stage parsing.
        
        Parameters
        ----------
        loader : callable

            a function to load the plugin

    """
    def __init__(self, name, *largs, **kwargs):
        kwargs['formatter_class'] = RawTextHelpFormatter
        kwargs['fromfile_prefix_chars']="@"
        args = kwargs.pop('args', None)
            
        preparser = BaseArgumentParser(add_help=False, 
                fromfile_prefix_chars=kwargs['fromfile_prefix_chars'])
        preparser.add_argument("-X", type=load, action="append")
        # Process the plugins
        preparser.exit = lambda a, b: None
#        preparser.convert_arg_line_to_args = self.convert_arg_line_to_args
        preparser._read_args_from_files = ArgumentParser._read_args_from_files.__get__(preparser)         
        preparser._yield_args_from_files = ArgumentParser._yield_args_from_files.__get__(preparser)         
        preparser.convert_args_file_to_args = ArgumentParser.convert_args_file_to_args.__get__(preparser)         

        self.ns, unknown = preparser.parse_known_args(args) 

        BaseArgumentParser.__init__(self, name, *largs, **kwargs)

        self.add_argument("-X", action='append', help='path of additional plugins to be loaded' )
 
    def parse_args(self, args=None):
        return BaseArgumentParser.parse_args(self, args)

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

from argparse import RawTextHelpFormatter

class HelpFormatterColon(RawTextHelpFormatter):
    """ This class is used to format the ':' seperated usage strings """
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = 'usage: '

        # this stripped down version supports no groups
        assert len(groups) == 0

        prog = '%(prog)s' % dict(prog=self._prog)

        # split optionals from positionals
        optionals = []
        positionals = []
        for action in actions:
            if action.option_strings:
                optionals.append(action)
            else:
                positionals.append(action)

        # build full usage string
        format = self._format_actions_usage
        action_usage = format(positionals + optionals, groups)
        usage = ''.join([s for s in [prog, action_usage] if s])
        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)

    def _format_actions_usage(self, actions, groups):
        # collect all actions format strings
        parts = []
        for i, action in enumerate(actions):

            # produce all arg strings
            if not action.option_strings:
                part = self._format_args(action, action.dest)

                part = ':' + part

                # add the action string to the list
                parts.append(part)

            # produce the first way to invoke the option in brackets
            else:
                option_string = action.option_strings[0]

                # if the Optional doesn't take a value, format is:
                #    -s or --long
                if action.nargs == 0:
                    part = '%s' % option_string

                # if the Optional takes a value, format is:
                #    -s ARGS or --long ARGS
                else:
                    default = action.dest.upper()
                    args_string = self._format_args(action, default)
                    part = '%s %s' % (option_string, args_string)

                # make it look optional if it's not required or in a group
                if not action.required:
                    part = '[:%s]' % part

                # add the action string to the list
                parts.append(part)

        # join all the action items with spaces
        text = ''.join([item for item in parts if item is not None])

        # return the text
        return text

