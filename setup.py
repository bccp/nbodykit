from setuptools import setup, find_packages
from glob import glob
import os
import re
import sys
import warnings

### versioning framework borrowed from xarray
MAJOR = 0
MINOR = 1
MICRO = 2
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git', 'git.cmd']:
        try:
            pipe = subprocess.Popen(
                [cmd, "describe", "--always", "--match", "v[0-9]*"],
                stdout=subprocess.PIPE)
            (so, serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if os.path.exists('nbodykit/version.py'):
            warnings.warn("WARNING: Couldn't get git revision, using existing xarray/version.py")
            write_version = False
        else:
            warnings.warn("WARNING: Couldn't get git revision, using generic version string")
    else:
        # have git, in git dir, but may have used a shallow clone (travis does this)
        rev = so.strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}", rev):
            # partial clone, manually construct version string
            # this is the format before we started using git-describe
            # to get an ordering on dev version strings.
            rev = "v%s.dev-%s" % (VERSION, rev)

        # Strip leading v from tags format "vx.y.z" to get th version string
        FULLVERSION = rev.lstrip('v')

else:
    FULLVERSION += QUALIFIER

def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'nbodykit', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

if write_version:
    write_version_py()
    
setup(name="nbodykit", 
      version=VERSION,
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

