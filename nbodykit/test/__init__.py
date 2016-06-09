import os
import sys
import unittest
import pytest

# idea borrowed from xarray
PY3 = sys.version_info[0] >= 3
if PY3:
    from urllib.request import urlretrieve as _urlretrieve
else:
    from urllib import urlretrieve as _urlretrieve
    
from nbodykit import examples_dir
cache_dir = os.path.join('~', '.nbodykit')

def download_data(github_url, cache_dir):
    """
    Download the github url tarball to the cache directory
    """
    import tarfile
    
    # make the cache dir if it doesnt exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # download the tarball locally
    tarball_link = os.path.join(github_url, 'tarball', 'master')
    tarball_local = os.path.join(cache_dir, 'master.tar.gz')
    _urlretrieve(tarball_link, tarball_local)
    
    # extract the tarball to the cache dir
    with tarfile.open(tarball_local) as tar:
    
        members = tar.getmembers()
        topdir = members[0].name
    
        for m in members[1:]:
            name = os.path.relpath(m.name, topdir)
            m.name = name
            tar.extract(m, path=cache_dir)
            
    # remove the downloaded tarball file
    if os.path.exists(tarball_local):
        os.remove(tarball_local)


def verify_data_in_cache(name, cache_dir=cache_dir,
                            github_url='https://github.com/bccp/nbodykit-data'):
    """
    Load a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the netcdf file containing the dataset
        ie. 'air_temperature'
    cache_dir : string, optional
        The directory in which to search for and write cached data.
    github_url : string
        Github repository where the data is stored
    """
    cache_dir = os.path.expanduser(cache_dir)
    longdir = os.path.join(cache_dir, 'data')
    localfile = os.path.join(longdir, name)

    # download and cache the nbodykit-data directory
    if not os.path.exists(localfile):
        download_data(github_url, cache_dir)
        
        # crash, if still no data
        if not os.path.exists(localfile):
            raise ValueError("filename `%s` does not exist in nbodykit-data directory" %name)
