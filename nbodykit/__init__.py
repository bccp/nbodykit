import sys
import os.path

def addpath(relpath):
    """
    Add path relative to root

    Parameters
    ----------
    relpath : string
        path to add

    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(root, relpath))

addpath('extern/kdcount')
addpath('extern/MP-sort')
addpath('extern/pfft-python')
addpath('extern/pypm')
addpath('extern/sharedmem')

import kdcount
import mpsort
import pfft
import pypm
import sharedmem
