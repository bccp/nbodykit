from __future__ import print_function
import argparse
import os
from jinja2 import Environment, FileSystemLoader, Template
import tempfile
import subprocess
import numpy

# the directory this file lives in
toplevel = os.path.split(os.path.abspath(__file__))[0]

def minutes_to_job_time(minutes):
    h, m = divmod(minutes, 60)
    return "%02d:%02d:00" % (h, m)

def get_nodes_from_cores(cores, host):
    if host == 'cori':
        nodes, extra = divmod(cores, 32)
    elif host == 'edison':
        nodes, extra = divmod(cores, 24)
    else:
        raise ValueError("bad host name '%s'" %host)
    # account for remainder cores
    if extra > 0: nodes += 1
    return nodes

class NERSCBenchmark(object):
    """
    Class to submit benchmark tasks to NERSC.
    """

    @staticmethod
    def load(path):
        """
        Return a NERSC benchmark result as a pandas dataframe.

        Parameters
        ----------
        path : str
            the path to a directory holding the results; this directory
            should hold a ``config.json`` file.
        """
        import pandas as pd

        # make sure config file exists first
        cfg_file = os.path.join(path, 'config.json')
        if not os.path.exists(cfg_file):
            raise ValueError("the path should contain a 'config.json' file")

        # load the config so we can attach to dataframe
        config = json.load(open(cfg_file, 'r'))

        # walk the directory and load .json files
        rows = []
        for dirpath, dirnames, filenames in os.walk(path):

            # check all files for .json extension
            for filename in filenames:

                if filename != 'config.json' and filename.endswith('.json'):

                    # first path is sample, if it exists
                    sample = os.path.normpath(dirpath).split(os.path.sep)[0]
                    if sample not in ['boss_like', 'desi_like']:
                        sample = ""

                    # load the JSON file
                    d = json.load(open(os.path.join(dirpath, filename), 'r'))

                    # add a row for each tagged benchmark in each test function
                    for test in d['tests']:
                        for tag in d['tags']:
                            x = d[test][tag]
                            row = (sample, test, tag, len(x))
                            row += (numpy.median(x), numpy.mean(x), numpy.min(x), numpy.max(x))
                            rows.append(row)

        # make the data frame
        names = ['sample', 'name', 'tag', 'ncores', 'dt_median', 'dt_mean', 'dt_min', 'dt_max']
        df = pd.DataFrame(data=rows, columns=names)
        df.config = config
        return df

    @classmethod
    def get_parser(cls):

        desc = 'run the specified benchmark on NERSC'
        parser = argparse.ArgumentParser(description=desc)

        h = 'the path of the benchmark to run in the nbodykit source code'
        parser.add_argument('benchname', type=str, help=h)

        h = 'the name of the sample to run'
        parser.add_argument('--sample', type=str, default=None, choices=['boss_like', 'desi_like'], help=h)

        h = 'the output path to save the benchmark result to'
        parser.add_argument('--bench-dir', type=str, help=h, required=True)

        h = 'the Python version to use'
        choices = ['2.7', '3.5', '3.6']
        parser.add_argument('--py', type=str, choices=choices, default='3.6', help=h)

        h = 'the number of nodes to request'
        parser.add_argument('-n', '--cores', type=int, help=h, required=True)

        h = 'the NERSC partition to submit to'
        choices=['debug', 'regular']
        parser.add_argument('-p', '--partition', type=str, choices=choices, default='debug', help=h)

        h = 'the requested amount of time (in minutes)'
        parser.add_argument('-t', '--time', type=int, default=30, help=h)

        h = 'the nbodykit source tag to checkout before running'
        parser.add_argument('--tag', type=str, default='master', help=h)

        return parser

    @classmethod
    def submit(cls):
        """
        Submit the benchmarking job on NERSC.
        """
        parser = cls.get_parser()
        ns = parser.parse_args()
        cls._submit(ns)

    @staticmethod
    def _submit(ns):
        """
        Internal function responsible for submiting the task, using the
        parsed command-line Namespace ``ns``.
        """
        # determine the NERSC host
        host = os.environ.get('NERSC_HOST', None)
        if host is None:
             raise RuntimeError("benchmark.py should be executed on NERSC")

        # make the benchmark directory absolute
        ns.bench_dir = os.path.normpath(os.path.abspath(ns.bench_dir))

        #  compute the job name and job script output path
        output_dir = os.path.join(ns.bench_dir, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = "{host}-%j.out.{cores}".format(host=host, cores=ns.cores)
        output_file = os.path.join(output_dir, output_file)

        # load the template job script
        jinja_env = Environment(loader=FileSystemLoader(toplevel))
        tpl = jinja_env.get_template('job.template.sh')

        # the configuration to pass to the template
        config = {}
        config['benchname'] = ns.benchname
        config['benchdir'] = ns.bench_dir
        config['partition'] = ns.partition
        config['time'] = minutes_to_job_time(ns.time)
        config['python_version'] = ns.py
        config['tag'] = ns.tag
        config['cores'] = ns.cores
        config['nodes'] = get_nodes_from_cores(ns.cores, host)
        config['haswell_config'] = "#SBATCH -C haswell" if host == 'cori' else ""
        config['output_file'] = output_file
        config['job'] = "benchmark.py"

        # the option to select specific sample
        if ns.sample is not None:
            config['sample'] = "-m %s" %ns.sample
        else:
            config['sample'] = ""

        # render the template
        rendered = tpl.render(**config)

        # write to temp file and call
        with tempfile.NamedTemporaryFile(mode='w') as ff:

            # echo the job scripts
            print(rendered)

            # write to temp file (and rewind)
            ff.write(rendered)
            ff.seek(0)

            # and call
            subprocess.call(["sbatch", ff.name])

if __name__ == '__main__':
    NERSCBenchmark.submit()
