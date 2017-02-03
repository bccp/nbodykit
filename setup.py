from setuptools import setup, find_packages
from glob import glob

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

setup(name="nbodykit", 
      version=find_version("nbodykit/version.py"),
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

