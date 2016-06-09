#! /usr/bin/env python

from nbodykit.test import cache_dir, download_data
import os

# download data from `nbodykit-data`
github_url='https://github.com/bccp/nbodykit-data'

# target directory
targetdir = os.path.expanduser(cache_dir)

# remove the old cache
if os.path.exists(targetdir):
    os.rmdir(targetdir)

# download
download_data(github_url, targetdir)