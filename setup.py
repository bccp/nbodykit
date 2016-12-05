from distutils.core import setup
from distutils.util import convert_path

import codecs
from glob import glob
import os
import re
import sys
import warnings

def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    here = os.path.abspath(os.path.dirname(__file__))
    return codecs.open(os.path.join(here, *parts), 'r').read()

def find_packages(base_path):
    base_path = convert_path(base_path)
    found = []
    for root, dirs, files in os.walk(base_path, followlinks=True):
        dirs[:] = [d for d in dirs if d[0] != '.' and d not in ('ez_setup', '__pycache__')]
        relpath = os.path.relpath(root, base_path)
        parent = relpath.replace(os.sep, '.').lstrip('.')
        if relpath != '.' and parent not in found:
            # foo.bar package but no foo package, skip
            continue
        for dir in dirs:
            if os.path.isfile(os.path.join(root, dir, '__init__.py')):
                package = '.'.join((parent, dir)) if parent else dir
                found.append(package)
    return found

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(name="nbodykit", 
      version=find_version("nbodykit", "__init__.py"),
      author="Yu Feng, Nick Hand, et al",
      maintainer="Yu Feng",
      maintainer_email="rainwoodman@gmail.com",
      description="Data analysis of cosmology simulations in parallel.",
      url="http://github.com/bccp/nbodykit",
      zip_safe=False,
      package_dir = {'nbodykit': 'nbodykit'},
      packages = find_packages('.'),
      install_requires=[
                'numpy', 'scipy', 'astropy',
                'mpi4py', 'mpi4py_test',
                'pmesh',
                'classylss',
                'kdcount',
                'mpsort',
                'bigfile'],
)

