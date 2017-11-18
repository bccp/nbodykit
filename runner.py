from argparse import ArgumentParser, Action, SUPPRESS
import os
import textwrap as tw
import itertools

def parametrize(params):
    """
    Execute a function for each of the input parameters.
    """
    keys = list(params.keys())
    params = list(itertools.product(*[params[k] for k in params]))
    def wrapped(func):
        def func_wrapper(*args, **kwargs):
            for p in params:
                kwargs.update(dict(zip(keys, p)))
                func(*args, **kwargs)

        return func_wrapper
    return wrapped

class InfoAction(Action):
    """
    Action similar to ``help`` to print out
    the various registered commands for the ``BenchmarkRunner``.
    class
    """
    def __init__(self,
                 option_strings,
                 dest=SUPPRESS,
                 default=SUPPRESS,
                 help=None):
        super(InfoAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):

        print("Registered commands\n" + "-"*20)
        for i, command in enumerate(BenchmarkRunner.commands):
            tag = BenchmarkRunner.tags[i]
            c = tw.dedent(command).strip()
            c = tw.fill(c, initial_indent=' '*4, subsequent_indent=' '*4, width=80)

            header = "%d:" %i
            if len(tag):
                header = header + " " + ", ".join(["'%s' = %s" %(k,tag[k]) for k in tag])
            header += "\n"
            print("%s\n%s\n" %(header, c))

        parser.exit()

class BenchmarkRunner(object):
    """
    Class to run ``benchmark.py`` commands.
    """
    commands = []
    tags = []

    @classmethod
    def register(cls, command, tag={}):
        """
        Register a new command.
        """
        cls.commands.append(command)
        cls.tags.append(tag)

    @classmethod
    def execute(cls):
        """
        Execute the ``BenchmarkRunner`` command.
        """
        # parse and get the command
        ns, unknown = cls.parse_args()

        # append unknown command-line args
        command = cls.commands[ns.testno] + ' ' + ' '.join(unknown)

        # execute
        cls._execute(command)

    @classmethod
    def _execute(cls, command):
        """
        Internal function to execute the command.

        Parameters
        ----------
        command : str
            the command to execute
        """
        # print the command
        c = tw.dedent(command).strip()
        c = tw.fill(c, initial_indent=' '*4, subsequent_indent=' '*4, width=80)
        print("executing:\n%s" %c)

        # execute
        os.system(command)

    @classmethod
    def parse_args(cls):
        """
        Parse the command-line arguments
        """
        desc = "run the ``benchmarks.py`` script from a set of registered commands"
        parser = ArgumentParser(description=desc)

        h = 'the integer number of the test to run'
        parser.add_argument('testno', type=int, help=h)

        h = 'print out the various commands'
        parser.add_argument('-i', '--info', action=InfoAction, help=h)

        ns, unknown = parser.parse_known_args()

        # make sure the integer value is valid
        if not (0 <= ns.testno < len(cls.commands)):
            N = len(cls.commands)
            raise ValueError("input ``testno`` must be between [0, %d]" %N)

        return ns, unknown
