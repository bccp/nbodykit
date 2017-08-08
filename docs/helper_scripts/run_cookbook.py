from __future__ import print_function
from glob import glob
import argparse
import os
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(filename):

    run_path = os.path.split(filename)[0]
    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': run_path}})
    with open(filename, 'wt') as f:
        nbformat.write(nb, f)


if __name__ == '__main__':

    desc = 'execute the Jupyter notebooks in the cookbook directory'
    parser = argparse.ArgumentParser(description=desc)

    h = 'which notebooks to execute; if none provided, all will be executed'
    parser.add_argument('filenames', nargs='*', default=[], help=h)

    ns = parser.parse_args()

    if not len(ns.filenames):
        dirname = os.path.split(__file__)[0]
        pattern = os.path.join(dirname, '..', 'source', 'cookbook', '*ipynb')
        ns.filenames = glob(pattern)

    for filename in ns.filenames:
        print("executing %s..." % os.path.split(filename)[-1])
        start = time.time()
        run_notebook(filename)
        stop = time.time()
        print("  ...done in %d seconds" %(stop-start))
