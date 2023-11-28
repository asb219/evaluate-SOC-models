"""
This package is organized into three submodules:

* :py:mod:`basefile<data_manager.file.basefile>`
  defines basic file types.
* :py:mod:`datafile<data_manager.file.datafile>`
  creates an interface between data files and
  :py:mod:`pandas` / :py:mod:`xarray` objects.
* :py:mod:`filefrom<data_manager.file.filefrom>`
  provides methods to download files from the Internet
  or extract them from archives.
"""

from .basefile import *
from .filefrom import *
from .datafile import *
