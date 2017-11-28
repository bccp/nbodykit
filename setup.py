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

def find_test_data(base_path, pkg_name):
    data = []
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d == 'data' and root.split('/')[-1] == 'tests':
                path = os.path.join(root, 'data', '*')
                data.append(os.path.relpath(path, pkg_name))
    return data

# the base dependencies
with open('requirements.txt', 'r') as fh:
    dependencies = [l.strip() for l in fh]

# extra dependencies
extras = {}
with open('requirements-extras.txt', 'r') as fh:
    extras['extras'] = [l.strip() for l in fh][1:]
    extras['full'] = extras['extras'] #

# package data
data = find_test_data('nbodykit', 'nbodykit')
data.append('style/*mplstyle') # also install mplstyle files

setup(name="nbodykit",
      version=find_version("nbodykit/version.py"),
      author="Yu Feng, Nick Hand, et al",
      maintainer="Yu Feng",
      maintainer_email="rainwoodman@gmail.com",
      description="Analysis kit for large-scale structure datasets, the massively parallel way",
      url="http://github.com/bccp/nbodykit",
      zip_safe=False,
      package_dir = {'nbodykit': 'nbodykit'},
      package_data = {'nbodykit': data},
      packages = find_packages('.'),
      license='GPL3',
      install_requires=dependencies,
      extras_require=extras
)
