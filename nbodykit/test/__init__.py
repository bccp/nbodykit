import os
import sys
import unittest
import pytest

from nbodykit.extern.six.moves.urllib import request
from nbodykit import examples_dir

def user_cache_dir(appname):
    r"""

    This function is copied from:
    https://github.com/pypa/pip/blob/master/pip/utils/appdirs.py

    Return full path to the user-specific cache dir for this application.
    
    Parameters
    ----------
    appname : str 
        the name of application
    
    Notes
    -----
    Typical user cache directories are:
        
        - Mac OS X: ~/Library/Caches/<AppName>
        - Unix: ~/.cache/<AppName> (XDG default)
        - Windows:  C:\Users\<username>\AppData\Local\<AppName>\Cache
    
    On Windows the only suggestion in the MSDN docs is that local settings go
    in the `CSIDL_LOCAL_APPDATA` directory. This is identical to the
    non-roaming app data dir (the default returned by `user_data_dir`). Apps
    typically put cache data somewhere *under* the given dir here. 
    
    Some examples:
    
        ...\Mozilla\Firefox\Profiles\<ProfileName>\Cache
        ...\Acme\SuperApp\Cache\1.0
    """

    from os.path import expanduser
    WINDOWS = (sys.platform.startswith("win") or
               (sys.platform == 'cli' and os.name == 'nt'))

    if WINDOWS:
        # Get the base path
        path = os.path.normpath(_get_win_folder("CSIDL_LOCAL_APPDATA"))

        # Add our app name and Cache directory to it
        path = os.path.join(path, appname, "Cache")
    elif sys.platform == "darwin":
        # Get the base path
        path = expanduser("~/Library/Caches")

        # Add our app name to it
        path = os.path.join(path, appname)
    else:
        # Get the base path
        path = os.getenv("XDG_CACHE_HOME", expanduser("~/.cache"))

        # Add our app name to it
        path = os.path.join(path, appname)

    return path


cache_dir = user_cache_dir('nbodykit')

def download_results_file(filename, localdir, 
                            github_url='https://github.com/bccp/nbodykit-data'):
    """
    Download a specific results file from the github repo
    """
    local_path = os.path.join(localdir, filename)
    if not os.path.exists(localdir):
        os.makedirs(localdir)
    
    # download the file
    if not os.path.exists(local_path):
        remote_path = os.path.join(github_url, 'raw', 'master', 'results', filename)
        request.urlretrieve(remote_path, local_path)
    
def download_data(github_user, github_repo, cache_dir):
    """
    Download the github url tarball to the cache directory
    """
    import tarfile

    # make the cache dir if it doesnt exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # download the tarball locally
    tarball_link = "https://codeload.github.com/%s/%s/legacy.tar.gz/master" %(github_user, github_repo)
    tarball_local = os.path.join(cache_dir, 'master.tar.gz')
    request.urlretrieve(tarball_link, tarball_local)
    
    if not tarfile.is_tarfile(tarball_local):
        dir_exists = os.path.exists(os.path.dirname(tarball_local))
        args = (tarball_local, str(dir_exists))
        raise ValueError("downloaded tarball '%s' cannot be opened as a tar.gz file (directory exists: %s)" %args)
    
    # extract the tarball to the cache dir
    with tarfile.open(tarball_local, 'r:*') as tar:

        members = tar.getmembers()
        topdir = members[0].name

        for m in members[1:]:
            name = os.path.relpath(m.name, topdir)
            m.name = name
            tar.extract(m, path=cache_dir)

    # remove the downloaded tarball file
    if os.path.exists(tarball_local):
        os.remove(tarball_local)


def verify_data_in_cache(name, cache_dir=cache_dir):
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
    """
    cache_dir = os.path.expanduser(cache_dir)
    longdir = os.path.join(cache_dir, 'data')
    localfile = os.path.join(longdir, name)

    # download and cache the nbodykit-data directory
    if not os.path.exists(localfile):
        download_data('bccp', 'nbodykit-data', cache_dir)

        # crash, if still no data
        if not os.path.exists(localfile):
            raise ValueError("filename `%s` does not exist in nbodykit-data directory" %name)
