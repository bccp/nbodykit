"""
    batchpower.py
    designed to use xargs to run batch jobs of 
    nbodykit power.py on NERSC
"""
import argparse as ap
import os
import tempfile
import subprocess

desc = "designed to use xargs to run batch jobs of nbodykit's power.py"
parser = ap.ArgumentParser(description=desc, 
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
                            
h = 'the name of the template argument file for power.py'
parser.add_argument('param_file', type=str, help=h)
h = 'the command to use; everything up until `power.py @params`'
parser.add_argument('command', help=h)
h = 'output all files to this directory'
parser.add_argument('-o', '--output', default=None, type=str, help=h)
h = 'the value to start the counter at (required)'
parser.add_argument('--start', default=0, type=int, help=h)
h = 'the value to stop the counter at'
parser.add_argument('--stop', required=True, type=int, help=h)
h = 'the increment of the counter'
parser.add_argument('--increment', default=1, type=int, help=h)
h = 'the number of xargs parallel processes to spawn'
parser.add_argument('-p',dest='xargs_nprocs',default=1, type=int, help=h)
h = 'replace occurences of `cnt-name` in template parameter file, using ' + \
    'string.format, with the value of the counter'
parser.add_argument('--cnt-name', default='cnt', type=str, help=h)
h = 'replace occurences of `output-name` in template parameter file, using ' + \
    'string.format, with the `output` option'
parser.add_argument('--output-name', default='output', type=str, help=h)

args = parser.parse_args()


def main():

    # read the template parameter file
    param_file = open(args.param_file, 'r').read()
    
    # generate the temporary parameter files
    tempfiles = []
    for cnt in range(args.start, args.stop, args.increment):
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            tempfiles.append(ff.name)
            kwargs = {args.cnt_name: cnt}
            if args.output is not None:
                kwargs[args.output_name] = args.output
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