from setuptools import setup, find_packages

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

setup(name="nbodykit", version="0.1pre",
      author="Yu Feng, Nick Hand, et al",
      maintainer="Yu Feng",
      maintainer_email="rainwoodman@gmail.com",
      description="Data analysis of cosmology simulations in parallel.",
      url="http://github.com/bccp/nbodykit",
      zip_safe=False,
      package_dir = {'nbodykit': 'nbodykit'},
      packages = find_packages(),
      install_requires=['numpy'],
      requires=['sharedmem', 'pypm', 'pfft', 'kdcount', 'mpsort'],
      cmdclass = {
        "build_py":build_py,
      }
)

