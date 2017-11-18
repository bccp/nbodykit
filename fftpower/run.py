#! /usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from runner import BenchmarkRunner, parametrize

# extract info from current path
this_dir = os.path.normpath(os.path.abspath(os.path.split(__file__)[0]))
results_dir = os.path.join(this_dir, 'results')
algorithm = this_dir.split(os.path.sep)[-1]

# CONFIGURATION
CORES = [64, 128, 256, 512]
SAMPLES = ['boss', 'desi']
TESTNAMES = ['test_strong_scaling']

@parametrize({'sample': SAMPLES, 'testname':TESTNAMES, 'ncores':CORES})
def add_commands(sample, testname, ncores):

    # the name of the benchmark test to run
    bench_name = os.path.join('benchmarks', algorithm, 'test_' + algorithm + '.py')
    bench_name += "::" + testname

    # the output directory
    bench_dir = os.path.join(results_dir, sample, ncores)

    # make the command
    command = f"python ../benchmark.py {bench_name} --sample {sample} --bench-dir {bench_dir} -n {ncores}"

    # and register
    tag = {'sample':sample, 'testname':testname, 'ncores':ncores}
    BenchmarkRunner.register(command, tag=tag)


if __name__ == '__main__':
    add_commands()
    BenchmarkRunner.execute()
