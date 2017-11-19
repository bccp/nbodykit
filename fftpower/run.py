#! /usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from runner import BenchmarkRunner

# current directory
this_dir = os.path.abspath(os.path.dirname(__file__))

# CONFIGURATION
ncores = [1, 4, 16, 64, 128, 256, 512]
test_functions = ['test_strong_scaling']


if __name__ == '__main__':

    result_dir = os.path.join(this_dir, 'results') # output directory
    test_path = 'benchmarks/test_fftpower.py' # the test file we want to run

    # initialize and run
    runner = BenchmarkRunner(test_path, result_dir)
    runner.add_commands(test_functions, ncores)
    runner.execute()
