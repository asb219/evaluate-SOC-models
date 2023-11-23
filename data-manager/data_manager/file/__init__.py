"""
This package is organized into three submodules:

* :py:mod:`basefile<RICH_data.base.file.basefile>`
  defines basic file types.
* :py:mod:`datafile<RICH_data.base.file.datafile>`
  creates an interface between data files and
  :py:mod:`pandas` / :py:mod:`xarray` objects.
* :py:mod:`filefrom<RICH_data.base.file.filefrom>`
  provides methods to download files from the Internet
  or extract them from archives.
"""

from .basefile import *
from .filefrom import *
from .datafile import *
