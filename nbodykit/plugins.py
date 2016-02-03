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
    return filename

from argparse import ArgumentParser as BaseArgumentParser
from argparse import RawTextHelpFormatter, Namespace
from argparse import Action, SUPPRESS

import re
import sys

def ReadConfigFile(**types):
    """
    Return an `argparse.Action` subclass that reads parameters from a file
    using YAML syntax
    
    This is designed to be used with `plugins.ArgumentParser` and be
    called via the command-line with a `-c` or `--config` option
    
    The action uses the underlying ArgumentParser to:
        * infer default values
        * check if parameter values are consistent with `choices`
        * infer the `type` of each parameter
        * check if any required parameters are missing
        
    Parameters
    ----------
    types : keywords
        the keys should be names of parameters and the values are functions
        that cast the parameter values appropriately. the function should
        take one argument, which is the value read from file
    """
    import yaml
    
    class ConfigFileAction(Action):
            
        def __call__(self, parser, namespace, values, option_string=None):
            
            # make a new namespace
            ns = Namespace()
        
            # read the yaml config file
            config = yaml.load(open(values, 'r'))
            
            # set defaults, check required and choices
            argnames = set([a.dest for a in parser._actions])
            missing = []
            for a in parser._actions:
                
                # set the default, if not suppressed
                if a.default != SUPPRESS:
                    setattr(ns, a.dest, a.default)
                    
                # check for choices
                if a.dest in config: 
                    if a.choices is not None:
                        if config[a.dest] not in a.choices:
                            args = (a.dest, config[a.dest], ", ".join(["'%s'" %x for x in a.choices]))
                            raise ValueError("argument %s: invalid choice '%s' (choose from %s)" %args)
                    if a.dest not in types and a.type is not None:
                        types[a.dest] = a.type
                else:        
                    # track missing        
                    if a.required: missing.append(a.dest)
            
            # raise error if missing
            if len(missing):
                missing = "(%s)" % ", ".join("'%s'" %k for k in missing)
                args = (parser.format_usage(), parser.prog, missing)
                raise ValueError("%s\n\n%s: too few arguments, missing: %s" %args)
                        
            # set the values, casting if available
            for k in config:
                if k in argnames:
                    v = config[k]
                    if k in types: v = types[k](v)
                    setattr(ns, k, v)
            
            # config attr of main namespace is new namespace
            setattr(namespace, self.dest, ns)
                                                    
    return ConfigFileAction
    
def ConfigPreparser(*keys):
    class PreparseConfigAction(Action):
        """
        Open a config file, read parameters assuming a YAML format, and
        load any plugins specified via the `X` key
        """
        def __call__(self, parser, namespace, values, option_string=None):
            import yaml
            
            # read the yaml config file
            config = yaml.load(open(values, 'r'))
            
            # search for plugins
            plugins = []
            if 'X' in config:
                plugins = config['X']
                if isinstance(plugins, str):
                    plugins = [plugins]
                for plugin in plugins: load(plugin)
        
            # save for debugging purposes   
            if namespace.X is not None: 
                namespace.X += plugins
            else:
                namespace.X = plugins
            
            # also grab any other specified keys
            for key in keys:
                if key in config:
                    setattr(namespace, key, config[key])
                    
    return PreparseConfigAction

class ArgumentParser(BaseArgumentParser):
    """ 
    An argument parser that loads plugins before dropping to
    the second stage parsing
    
    Plugins can be specified on the command line with `-X`
    option or set from a YAML config file, which is
    passed via `-c` or `--config`. 
    """
    def __init__(self, name, preparse_from_config=[], *largs, **kwargs):

        # initialize the preparser
        kwargs['formatter_class'] = RawTextHelpFormatter
        kwargs['fromfile_prefix_chars']="@"

        # do the base initialization
        BaseArgumentParser.__init__(self, name, *largs, **kwargs)

        # parse -X on cmdline or search config file for -X options
        self.add_argument("-X", type=load, action="append")
        self.add_argument('-c', '--config', action=ConfigPreparser(*preparse_from_config))

        # track error messages
        self.error_messages = []
        
    def error(self, message):
        """
        If `config` is an option, ignore all exceptions so we can read 
        options from file, otherwise, use the default `error` behavior
        """
        self.error_messages.append(message)
        
    def parse_known_args(self, args=None, namespace=None):
        """
        If `config` is a option, return the namespace created from reading
        the YAML file -- otherwise, use the default behavior
        """
        if args is None:
            import sys
            args = sys.argv[1:]
        result = BaseArgumentParser.parse_known_args(self, args)
        
        # if parsing failed, result is None
        if result is None:
            if len(self.error_messages):
                BaseArgumentParser.error(self, self.error_messages[0])
        
        ns, args = result
        no_config = 'config' not in ns or ns.config is None
        
        # if no config file, check errors
        if no_config and len(self.error_messages):
            BaseArgumentParser.error(self, self.error_messages[0])
        
        # okay, no errors -- return either config or cmd-line namespace
        if no_config:
            return ns, args
        else:
            return ns.config, args

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

def ListPluginsAction(extensionpoint):
    class ListPluginsAction(Action):
        def __init__(self,
                     option_strings,
                     dest=SUPPRESS,
                     default=SUPPRESS,
                     help=None):
            Action.__init__(self, 
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                help=help)
        
        def __call__(self, parser, namespace, values, option_string=None):
            parser.exit(0, extensionpoint.format_help())
    return ListPluginsAction
