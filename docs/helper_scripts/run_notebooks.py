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

def sort_paths(paths):
    dirs = []; files = []
    for f in paths:
        path = os.path.normpath(os.path.abspath(f))
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            dirs.append(path)
        else:
            raise RuntimeError("no such path: '%s'" % path)
    return dirs, files


if __name__ == '__main__':

    desc = 'find and execute the Jupyter notebooks in the docs directory (but not the cookbook)'
    parser = argparse.ArgumentParser(description=desc)

    h = 'which notebooks to execute; if none provided, all will be executed'
    parser.add_argument('paths', nargs='*', default=[], help=h)

    ns = parser.parse_args()

    # execute all?
    execute_all = not len(ns.paths)

    # absolute paths of input and exclude paths
    input_dirs, input_files = sort_paths(ns.paths)

    # the list of notebooks to execute
    notebooks = []

    # walk the full directory path
    for dirpath, dirs, filenames in os.walk(os.path.join(thisdir, '..', 'source')):

        # exclude cookbook!!
        dirs[:] = [d for d in dirs if d != 'cookbook']

        # normalize the current dirpath
        dirpath = os.path.normpath(dirpath)

        # find the valid notebooks
        for f in filenames:
            if f.endswith('.ipynb') and 'checkpoint' not in f:
                f = os.path.join(dirpath, f) # the full path

                # see if this path was specified by user
                if not execute_all:
                    if dirpath not in input_dirs and f not in input_files:
                        continue
                notebooks.append(f)

    for filename in notebooks:
        print("executing %s..." % os.path.split(filename)[-1])
        start = time.time()
        run_notebook(filename)
        stop = time.time()
        print("  ...done in %d seconds" %(stop-start))
