#! /usr/bin/env python

from __future__ import print_function
from nbodykit.test import cache_dir, download_data
import os
import shutil
import argparse


def print_cache():
    """
    Print the cache directory to stdout
    """
    print("cache directory: ", os.path.expanduser(cache_dir))
    
def download_cache():
    """
    Download the cache directory, overwriting any current cache
    """
    # target directory
    targetdir = os.path.expanduser(cache_dir)
    
    # remove the old cache
    if os.path.exists(targetdir):
        print("removing old cache")
        shutil.rmtree(targetdir)

    # download
    download_data('bccp', 'nbodykit-data', targetdir)
    print("successfully downloaded `nbodykit-data` cache to '%s'" %targetdir)
    
def remove_cache():
    """
    Remove the cache directory
    """
    # target directory
    targetdir = os.path.expanduser(cache_dir)
    
    # remove the old cache
    if os.path.exists(targetdir):
        print("removing the existing cache at '%s'" %targetdir)
        shutil.rmtree(targetdir)
    else:
        print("no existing cache to remove")

if __name__ == '__main__':
    
    desc = "manage the cache: print directory path, download, clear, etc"
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(dest='subparser_name')
    
    # print cache_dir
    h = 'print the cache directory path'
    show = subparsers.add_parser('print', help=h)
    show.set_defaults(func=print_cache)
    
    # download
    h = 'download the cache directory, overwriting any current cache'
    download = subparsers.add_parser('download', help=h)
    download.set_defaults(func=download_cache)
    
    # remove
    h = 'remove the cache directory'
    remove = subparsers.add_parser('remove', help=h)
    remove.set_defaults(func=remove_cache)
    
    ns = parser.parse_args()
    ns.func()