from __future__ import print_function
import argparse
import os
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

thisdir = os.path.abspath(os.path.split(__file__)[0])

def run_notebook(filename):

    run_path = os.path.split(filename)[0]
    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': run_path}})
    with open(filename, 'wt') as f:
        nbformat.write(nb, f)


if __name__ == '__main__':

    desc = 'find and execute the Jupyter notebooks in the docs directory'
    parser = argparse.ArgumentParser(description=desc)

    h = 'which notebooks to execute; if none provided, all will be executed'
    parser.add_argument('filenames', nargs='*', default=[], help=h)
    ns = parser.parse_args()

    # absolute paths of input files
    input_filenames = [os.path.normpath(os.path.abspath(f)) for f in ns.filenames]

    notebooks = []
    for dirpath, dirs, filenames in os.walk(os.path.join(thisdir, '..', 'source')):

        # find the valid notebooks
        for f in filenames:
            if f.endswith('.ipynb') and 'checkpoint' not in f:
                f = os.path.normpath(os.path.join(dirpath, f)) # the full path
                if len(input_filenames) and f not in input_filenames:
                    continue
                notebooks.append(f)

    for filename in notebooks:
        print("executing %s..." % os.path.split(filename)[-1])
        start = time.time()
        run_notebook(filename)
        stop = time.time()
        print("  ...done in %d seconds" %(stop-start))
