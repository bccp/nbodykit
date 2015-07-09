"""
    batchpower.py
    designed to use xargs to run batch jobs of 
    nbodykit power.py on NERSC
"""
import argparse as ap
import os
import tempfile
import subprocess
        
class DictAction(ap.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        key, val = values.split(':')
        getattr(namespace, self.dest)[key.strip()] = val.strip()
        
desc = "designed to use xargs to run batch jobs of nbodykit's power.py"
parser = ap.ArgumentParser(description=desc, 
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
                            
h = 'the name of the template argument file for power.py'
parser.add_argument('param_file', type=str, help=h)
h = 'the command to use; everything up until `power.py @params`'
parser.add_argument('command', help=h)
h = 'the value to start the counter at'
parser.add_argument('--start', default=0, type=int, help=h)
h = 'the value to stop the counter at (required)'
parser.add_argument('--stop', required=True, type=int, help=h)
h = 'the increment of the counter'
parser.add_argument('--increment', default=1, type=int, help=h)
h = 'the number of xargs parallel processes to spawn'
parser.add_argument('-p',dest='xargs_nprocs',default=1, type=int, help=h)
h = 'replace occurences of `iter-str` in template parameter file, using ' + \
    'string.format, with the value of the counter'
parser.add_argument('--iter-str', default='cnt', type=str, help=h)
h = 'extra string replaces, specified as `output:output a:0.6452`. ' + \
    'Values assumed to be strings, unless they match another command line argument'
parser.add_argument('--extra-str-fmts', '-X', dest='extras', action=DictAction, help=h)

# parse
args = parser.parse_args()

def main():

    # read the template parameter file
    param_file = open(args.param_file, 'r').read()

    # generate the temporary parameter files
    tempfiles = []
    for cnt in range(args.start, args.stop, args.increment):
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            tempfiles.append(ff.name)
            kwargs = {args.iter_str : cnt}
            if args.extras is not None:
                for k in args.extras:
                    kwargs[k] = args.extras[k]
            ff.write(param_file.format(**kwargs))

    # echo the names of the tempfiles
    echo = subprocess.Popen(["echo"] + ['\n'.join(tempfiles)], stdout=subprocess.PIPE)

    # form the xargs
    xargs_command = ['xargs', '-P', str(args.xargs_nprocs), '-n', '1', '-I', '%']
    xargs_command += args.command.split() + ['power.py', '@%']

    # try to do the call
    try:
        xargs = subprocess.Popen(xargs_command, stdin=echo.stdout)
        echo.stdout.close()
        xargs.communicate()
        echo.wait()
    except:
        pass
    finally:
        # delete the temporary files
        for tfile in tempfiles:
            if os.path.exists(tfile):
                os.remove(tfile)


if __name__ == '__main__':
    main()