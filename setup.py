from distutils.core import setup
from distutils.util import convert_path
import os

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

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

setup(name="nbodykit", 
      version=find_version("nbodykit/version.py"),
      author="Yu Feng, Nick Hand, et al",
      maintainer="Yu Feng",
      maintainer_email="rainwoodman@gmail.com",
      description="Analysis kit for large-scale structure datasets, the massively parallel way",
      url="http://github.com/bccp/nbodykit",
      zip_safe=False,
      package_dir = {'nbodykit': 'nbodykit'},
      packages = find_packages('.'),
      install_requires=[
                'numpy', 'scipy', 'astropy',
                'mpi4py', 'mpi4py_test',
                'pmesh',
                'kdcount',
                'mpsort',
                'bigfile', 'dask'],
)

