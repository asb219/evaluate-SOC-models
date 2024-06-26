"""
Paths of important locations such as model input and output directories.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

from pathlib import Path
from evaluate_SOC_models.config import get_config

config = get_config()

PACKAGEPATH = Path(__file__).resolve().parent

def absolute_path(x):
    return ( PACKAGEPATH / Path(config['path'][x]).expanduser() ).resolve()

REPOSITORYPATH = absolute_path('repository')
MENDREPOSITORYPATH = REPOSITORYPATH / 'MEND'
MIMICS2021REPOSITORYPATH = REPOSITORYPATH / 'MIMICS2021'
DUMPPATH = absolute_path('dump')
DOWNLOADPATH = absolute_path('downloads')
DATAPATH = absolute_path('data')
RESULTSPATH = absolute_path('results')
LOGPATH = absolute_path('log')
LOGFILEPATH = LOGPATH / config['log']['filename']

TOPSOIL_MIN_DEPTH = int(config['topsoil']['min_depth'])
TOPSOIL_MAX_DEPTH = int(config['topsoil']['max_depth'])
SAVEPATH = RESULTSPATH / \
    f'topsoil_min{TOPSOIL_MIN_DEPTH}cm_max{TOPSOIL_MAX_DEPTH}cm_depth'
SAVEOUTPUTPATH = SAVEPATH / 'model-output'
SAVEALLDATAPATH = SAVEPATH / 'data'

del Path, get_config, config, absolute_path
