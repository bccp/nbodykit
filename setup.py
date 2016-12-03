from setuptools import setup, find_packages
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
      package_data = {'nbodykit': list(glob('nbodykit/plugins/*/*.py'))},
      include_package_data=True,
      packages = find_packages(),
      install_requires=['numpy'],
      requires=['sharedmem', 'pmesh', 'pfft', 'kdcount', 'mpsort', 'scipy', 'bigfile'],
      scripts=['bin/nbkit.py', 'bin/nbkit-batch.py'],
)

