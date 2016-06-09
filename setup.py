from setuptools import setup, find_packages
from glob import glob
setup(name="nbodykit", version="0.1.1",
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

